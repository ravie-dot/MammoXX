"""
Py2app setup for MammoX Medical Annotation Tool
"""
from setuptools import setup

APP = ['main.py']
DATA_FILES = [
    ('', ['mammo_paths.py']),
]

OPTIONS = {
    'argv_emulation': True,
    'packages': ['PyQt5', 'cv2', 'numpy', 'csv', 'json', 'pathlib'],
    'includes': [
        'PyQt5.QtCore', 
        'PyQt5.QtGui', 
        'PyQt5.QtWidgets',
        'mammo_paths'
    ],
    'excludes': [
        'matplotlib', 
        'scipy', 
        'pandas', 
        'tensorflow', 
        'torch',
        'jupyter',
        'IPython'
    ],
    'plist': {
        'CFBundleName': 'MammoX',
        'CFBundleDisplayName': 'MammoX Medical Annotation Tool',
        'CFBundleIdentifier': 'com.mammox.annotation',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': True,
    },
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)