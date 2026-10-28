import os
import sys
import hashlib

base_path = sys._MEIPASS

paths = [base_path,os.path.join(base_path, "bin")]

intel_marker = os.path.join(base_path, "ur_adapter_level_zero.dll")
if os.path.exists(intel_marker):
    system_path = os.environ.get("PATH", "")
    if system_path:
        paths.append(system_path)

os.environ["PATH"] = os.pathsep.join(paths)
os.environ["LADA_MODEL_WEIGHTS_DIR"] = os.path.join(base_path, "model_weights")
os.environ["LOCALE_DIR"] = os.path.join(base_path, "lada", "locale")

# GTK/GStreamer can block while reusing another portable installation's plugin
# registry. Keep an app-specific registry, keyed by runtime path and revision.
cache_root = os.path.join(os.environ.get('LOCALAPPDATA', base_path), 'lada', 'cache')
try:
    os.makedirs(cache_root, exist_ok=True)
    plugins = os.path.join(base_path, 'gst_plugins')
    revision = os.stat(plugins).st_mtime_ns if os.path.isdir(plugins) else 0
    registry_key = hashlib.sha256(f'{os.path.normcase(base_path)}:{revision}'.encode()).hexdigest()[:16]
    os.environ['GST_REGISTRY'] = os.path.join(cache_root, f'gst-registry-{registry_key}.bin')
    plugin_paths = [os.path.join(base_path, name) for name in ('gst_plugins', 'gst-plugins')
                    if os.path.isdir(os.path.join(base_path, name))]
    if plugin_paths:
        os.environ['GST_PLUGIN_PATH'] = os.pathsep.join(plugin_paths)
except OSError:
    pass  # Honor the platform default if the cache directory is not writable.
