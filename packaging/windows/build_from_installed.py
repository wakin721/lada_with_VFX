# SPDX-License-Identifier: AGPL-3.0
"""Compile this fork locally using an existing Windows Lada runtime.

The installed copy is read-only. A new portable folder is produced separately.
This preserves the installed PyTorch/GTK/native libraries; a clean dependency
build is still available via package_executable.ps1 and GitHub Actions.
Run with the same Python minor version as the installed bundle (currently 3.13).
Requires PyInstaller 6.18 and nvidia-vfx 0.2 in the build environment.
"""
import argparse
import importlib.metadata
import importlib.util
import marshal
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import zlib

from PyInstaller.archive.readers import CArchiveReader
from PyInstaller.archive.writers import CArchiveWriter, ZlibArchiveWriter
import pefile

COOKIE_MAGIC = b'MEI\014\013\012\013\016'
COOKIE_FORMAT = '!8sIIII64s'


def rebuild_exe(original, destination, pyz_path, main_source):
    data = original.read_bytes()
    cookie_pos = data.rfind(COOKIE_MAGIC)
    if cookie_pos < 0:
        raise ValueError(f'Not a supported PyInstaller executable: {original}')
    cookie = struct.unpack_from(COOKIE_FORMAT, data, cookie_pos)
    if cookie[4] != sys.version_info.major * 100 + sys.version_info.minor:
        raise ValueError(f'Bundle needs Python {cookie[4]}, builder is {sys.version_info[:2]}')
    start = cookie_pos + struct.calcsize(COOKIE_FORMAT) - cookie[1]
    toc_data = data[start + cookie[2]:start + cookie[2] + cookie[3]]
    payload = bytearray()
    toc = []
    offset = 0
    main_replaced = False
    while offset < len(toc_data):
        size, position, compressed, raw, flag, kind = struct.unpack_from('!iIIIBc', toc_data, offset)
        name = toc_data[offset + 18:offset + size].rstrip(b'\0').decode('utf-8')
        kind = kind.decode('ascii')
        entry = data[start + position:start + position + compressed]
        if kind == 'z':
            entry = pyz_path.read_bytes()
            raw, flag = len(entry), 0
        elif kind == 's' and name in ('main', 'pyinstaller_runtime_hook_lada'):
            source = main_source if name == 'main' else Path(__file__).with_name('pyinstaller_runtime_hook_lada.py').read_text(encoding='utf-8')
            code = compile(source, name + '.py', 'exec', optimize=0)
            raw_code = marshal.dumps(code)
            entry, raw, flag = zlib.compress(raw_code), len(raw_code), 1
            if name == 'main':
                main_replaced = True
        toc.append((len(payload), len(entry), raw, flag, kind, name))
        payload.extend(entry)
        offset += size
    if not main_replaced:
        raise ValueError('Entry point was not found; executable was not rebuilt')
    # PyInstaller's GStreamer hook otherwise overwrites the app-specific registry.
    runtime_hook = next((entry for entry in toc if entry[-1] == 'pyinstaller_runtime_hook_lada'), None)
    if runtime_hook:
        toc.remove(runtime_hook)
        toc.insert(next(i for i, entry in enumerate(toc) if entry[-1] == 'main'), runtime_hook)
    toc_bytes = CArchiveWriter._serialize_toc(toc)
    footer = struct.pack(COOKIE_FORMAT, COOKIE_MAGIC,
                         len(payload) + len(toc_bytes) + struct.calcsize(COOKIE_FORMAT),
                         len(payload), len(toc_bytes), cookie[4], cookie[5])
    destination.write_bytes(data[:start] + payload + toc_bytes + footer)
    pe = pefile.PE(str(destination), fast_load=True)
    pe.OPTIONAL_HEADER.CheckSum = pe.generate_checksum()
    pe.write(str(destination))
    pe.close()
    # Independently read the rebuilt archive before considering it a build.
    check = CArchiveReader(str(destination)).open_embedded_archive('PYZ.pyz')
    assert 'lada.utils.vfx' in check.toc and 'nvvfx' in check.toc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--installed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--refresh', action='store_true', help='Recompile an existing separate staging directory; retain its native runtime')
    parser.add_argument('--msgfmt', type=Path, help='Optional CPython Tools/i18n/msgfmt.py')
    parser.add_argument('--probe-source', type=Path, help='Build an additional GUI-runtime diagnostic executable')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    installed, output = args.installed.resolve(), args.output.resolve()
    if output.exists() and not args.refresh:
        parser.error('Output must be a new directory; the installed application is never overwritten here')
    if installed == output or installed in output.parents or output in installed.parents:
        parser.error('Output and installed directories must be separate')
    if not (installed / '_internal/python313.dll').is_file() or sys.version_info[:2] != (3, 13):
        parser.error('This runtime-reuse build requires an installed Python 3.13 Lada bundle and Python 3.13')
    if args.refresh and not (output / '_internal/python313.dll').is_file():
        parser.error('--refresh requires an existing staging runtime')
    vfx_distribution = importlib.metadata.distribution('nvidia-vfx')
    if vfx_distribution.version != '0.2.0.0':
        parser.error('This build targets nvidia-vfx==0.2.0.0')

    output.mkdir(parents=True, exist_ok=args.refresh)
    if not args.refresh:
        print('Copying the existing native runtime into a separate staging directory...', flush=True)
        result = subprocess.run(['robocopy', str(installed / '_internal'), str(output / '_internal'),
                                 '/E', '/COPY:DAT', '/R:1', '/W:1', '/NFL', '/NDL', '/NJH', '/NJS'])
        if result.returncode >= 8:
            raise RuntimeError(f'Runtime copy failed: robocopy exit {result.returncode}')

    entries = {}
    code_dict = {}
    # Reuse dependency bytecode from both original entry points. All Lada code
    # below is replaced with freshly compiled source from this checkout.
    for executable in ('lada-cli.exe', 'lada.exe'):
        pyz = CArchiveReader(str(installed / executable)).open_embedded_archive('PYZ.pyz')
        for name, item in pyz.toc.items():
            if name == 'lada' or name.startswith('lada.'):
                continue
            kind = item[0]
            if kind == 3:
                entries[name] = (name, None, 'PYMODULE')
            else:
                filename = '__init__.py' if kind == 1 else name.rsplit('.', 1)[-1] + '.py'
                entries[name] = (name, filename, 'PYMODULE')
                code_dict[name] = pyz.extract(name)

    package_roots = [(root / 'lada', 'lada'), (Path(vfx_distribution.locate_file('nvvfx')), 'nvvfx')]
    compiled_count = 0
    for package_root, package_name in package_roots:
        for path in sorted(package_root.rglob('*.py')):
            relative = path.relative_to(package_root)
            parts = relative.with_suffix('').parts
            is_package = parts[-1] == '__init__'
            module = '.'.join((package_name,) + (parts[:-1] if is_package else parts))
            entries[module] = (module, str(path), 'PYMODULE')
            code_dict[module] = compile(path.read_text(encoding='utf-8-sig'), str(path), 'exec', optimize=0)
            compiled_count += 1

    pyz_path = output.parent / (output.name + '-compiled.pyz')
    print(f'Compiling {compiled_count} source modules and assembling the application...', flush=True)
    ZlibArchiveWriter(str(pyz_path), sorted(entries.values()), code_dict)
    for executable, entry in [('lada-cli.exe', 'cli'), ('lada.exe', 'gui')]:
        rebuild_exe(installed / executable, output / executable, pyz_path,
                    (root / f'lada/{entry}/main.py').read_text(encoding='utf-8'))

    target_vfx = output / '_internal/nvvfx'
    shutil.copytree(package_roots[1][0], target_vfx, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    for file in vfx_distribution.files:
        if '.dist-info' in file.parts[0]:
            target = output / '_internal' / str(file)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(vfx_distribution.locate_file(file), target)
    for pattern in ('*.ui', '*.css', '*.gresource', '*.csv'):
        for path in (root / 'lada').rglob(pattern):
            target = output / '_internal' / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    if args.msgfmt:
        for lang in ('zh_CN', 'zh_TW'):
            target = output / f'_internal/lada/locale/{lang}/LC_MESSAGES/lada.mo'
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([sys.executable, str(args.msgfmt.resolve()), '-o', str(target),
                            str(root / f'translations/{lang}.po')], check=True)
    if args.probe_source:
        rebuild_exe(installed / 'lada.exe', output / 'vfx-build-probe.exe', pyz_path,
                    args.probe_source.read_text(encoding='utf-8'))
    print(f'Local build complete: {output}', flush=True)


if __name__ == '__main__':
    main()
