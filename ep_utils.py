import onnxruntime
import onnxruntime_ep_openvino as openvino_ep
from typing import List, Any
import os


# Global variables for plugin EP registration tracking
_plugin_ep_registered = False
_plugin_ep_name = None

class ColorLogger:
    """Color logging utility for consistent formatted output."""
    
    # ANSI color codes
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    DEBUG_ENABLED = False

    @staticmethod
    def set_debug_enabled(enabled: bool) -> None:
        """Globally enable/disable debug logging."""
        ColorLogger.DEBUG_ENABLED = bool(enabled)

    @staticmethod
    def is_debug_enabled() -> bool:
        return ColorLogger.DEBUG_ENABLED
    
    @staticmethod
    def success(message: str, indent: int = 0) -> None:
        """Print success message in green."""
        indent_str = " " * indent
        print(f"{indent_str}{ColorLogger.GREEN}{message}{ColorLogger.RESET}")
    
    @staticmethod
    def info(message: str, indent: int = 0) -> None:
        """Print info message in blue."""
        indent_str = " " * indent
        print(f"{indent_str}{ColorLogger.BLUE}{message}{ColorLogger.RESET}")
    
    @staticmethod
    def warning(message: str, indent: int = 0) -> None:
        """Print warning message in yellow."""
        indent_str = " " * indent
        print(f"{indent_str}{ColorLogger.YELLOW}{message}{ColorLogger.RESET}")
    
    @staticmethod
    def error(message: str, indent: int = 0) -> None:
        """Print error message in red."""
        indent_str = " " * indent
        print(f"{indent_str}{ColorLogger.RED}{message}{ColorLogger.RESET}")
    
    @staticmethod
    def debug(message: str, indent: int = 0, newline_before: bool = True) -> None:
        """Print debug message in yellow with optional leading newline.
        
        The newline_before option ensures debug output appears on a new line
        when printed after inline token output (which uses sys.stdout.write).
        """
        if not ColorLogger.DEBUG_ENABLED:
            return
        indent_str = " " * indent
        prefix = "\n" if newline_before else ""
        print(f"{prefix}{indent_str}{ColorLogger.YELLOW}DEBUG: {message}{ColorLogger.RESET}")
    
    @staticmethod
    def separator(title: str = "", width: int = 60, color: str = GREEN) -> None:
        """Print a formatted separator line."""
        if title:
            padding = (width - len(title) - 2) // 2
            separator_line = f"{'-' * padding} {title} {'-' * padding}"
            if len(separator_line) < width:
                separator_line += '-'
        else:
            separator_line = '-' * width
        
        print(f"{color}{separator_line}{ColorLogger.RESET}")
        
class OVEPManager:
    def __init__(self, device_type: str):
        self.ep_name_ = openvino_ep.get_ep_name()
        self.device_type = device_type

    def register_plugin_ep(self) -> None:
        """
        Register the OpenVINO Plugin EP for plugin (ABI) mode.
        """
        global _plugin_ep_registered, _plugin_ep_name      
        ep_name = self.ep_name_
        
        # Register the plugin EP library
        onnxruntime.register_execution_provider_library(ep_name, openvino_ep.get_library_path())
        _plugin_ep_registered = True
        _plugin_ep_name = ep_name      

    def get_plugin_ep_devices(self, ep_name: str, device_type: str) -> List[Any]:
        """
        Get EP devices matching the specified device type for PSU models.
        Supports single devices (NPU, CPU, GPU) and meta-devices (HETERO:NPU,CPU)
        
        This matches the C++ test reference (ort_abi_device_wrapper.h):
        if (ep_name == expected_ep_name && device_set.count(ov_device) == 1)
        
        EXACT match ensures:
        - For single device (target="OpenVINOExecutionProvider"):
            only base "OpenVINOExecutionProvider" matches; ".HETERO"/".AUTO" variants do NOT
        - For meta device (target="OpenVINOExecutionProvider.HETERO"):
            only "OpenVINOExecutionProvider.HETERO" matches; base EP does NOT
            This routes through the HETERO factory which builds "HETERO:NPU,CPU" correctly
        
        :param ep_name: Name of the execution provider (e.g., 'OpenVINOExecutionProvider')
        :param device_type: Device type to match (e.g., 'NPU', 'CPU', 'GPU', 'HETERO:NPU,CPU')
        :return: List of matching EP devices
        """
        meta_prefix = ""
        ov_device_types = []
        
        # Check for meta-device prefix (e.g. "HETERO:")
        if ':' in device_type:
            meta_prefix, remainder = device_type.split(':', 1)
            meta_prefix = meta_prefix.upper()
        else:
            remainder = device_type
        
        # Split comma-separated devices
        ov_device_types = [d.strip() for d in remainder.split(',') if d.strip()]
        
        # Get all EP devices
        ep_devices = onnxruntime.get_ep_devices()
        
        # Build device info list
        device_info_list = [f"{ep.ep_name} (ov_device: {ep.ep_metadata.get('ov_device', 'N/A')})" for ep in ep_devices]
        ColorLogger.info(f"Available EP devices: {device_info_list}")
        
        # Helper: match EP device by EXACT ep_name match and ov_device metadata
        # CRITICAL: Use EXACT match (==), not substring match (in)
        def get_ep_device(target_ep_name: str, ov_device: str):
            for ep_device in ep_devices:
                if ep_device.ep_name == target_ep_name:  # EXACT match
                    metadata = ep_device.ep_metadata
                    if 'ov_device' in metadata and metadata['ov_device'] == ov_device:
                        return ep_device
            return None
        
        # Build target EP name
        ep_name_with_meta = ep_name
        if meta_prefix:
            ep_name_with_meta = f"{ep_name}.{meta_prefix}"
        
        # For HETERO/AUTO meta-devices
        if meta_prefix:
            # For HETERO/AUTO mode: use the meta-device EP variant
            # e.g., "OpenVINOExecutionProvider.HETERO" rather than base "OpenVINOExecutionProvider"
            # The base EP is NOT a meta device — passing 2 base EPs causes the factory
            # to discard all but the first device and fail on HETERO-specific config (affinity).
            ColorLogger.info(f"Meta-device mode ({meta_prefix}): collecting '{ep_name_with_meta}' EPs for: {ov_device_types}")
            
            session_ep_devices = []
            for ov_device in ov_device_types:
                # Look for the HETERO-specific EP device (e.g., OpenVINOExecutionProvider.HETERO with ov_device=NPU)
                plugin_ep_device = get_ep_device(ep_name_with_meta, ov_device)
                
                # If not found and device has suffix (e.g., "GPU.0"), try without suffix
                if plugin_ep_device is None and '.' in ov_device:
                    base_device = ov_device.split('.')[0]
                    plugin_ep_device = get_ep_device(ep_name_with_meta, base_device)
                    if plugin_ep_device:
                        ColorLogger.info(f"Matched device '{ov_device}' to base device '{base_device}'")
                
                if plugin_ep_device is None:
                    ColorLogger.warning(f"Did not find an EP device with ep_name = '{ep_name_with_meta}' & ov_device = {ov_device}")
                    continue
                
                session_ep_devices.append(plugin_ep_device)
                ColorLogger.success(f"Found EP device: {plugin_ep_device.ep_name}, OV device: {plugin_ep_device.ep_metadata.get('ov_device', 'N/A')}")
            
            return session_ep_devices
        else:
            # For single device types (NPU, CPU, GPU), look for matching ov_device
            ColorLogger.info(f"Looking for EP device: {ep_name} with device type: {device_type}")
            
            # Use the first device type from the list
            target_device_type = ov_device_types[0] if ov_device_types else device_type
            plugin_ep_device = get_ep_device(ep_name, target_device_type)
            
            if plugin_ep_device:
                ColorLogger.success(f"Found EP device: {plugin_ep_device.ep_name}, OV device: {target_device_type}")
                return [plugin_ep_device]
            
            ColorLogger.warning(f"Did not find an EP device with ep_name = {ep_name} & ov_device = {target_device_type}")
            return []

    def unregister_plugin_ep(self) -> None:
            """
            Unregister the OpenVINO Plugin EP when done.
            Should be called when the session manager is no longer needed.
            """
            global _plugin_ep_registered, _plugin_ep_name
            
            if _plugin_ep_registered and _plugin_ep_name:
                try:
                    onnxruntime.unregister_execution_provider_library(_plugin_ep_name)
                    ColorLogger.info("Successfully unregistered Plugin EP")
                    _plugin_ep_registered = False
                    _plugin_ep_name = None
                except Exception as e:
                    ColorLogger.warning(f"Failed to unregister Plugin EP: {e}")