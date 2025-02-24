# -*- mode: python -*-

block_cipher = None


a = Analysis(['mainwindow.py'],
             pathex=['.'],
             binaries=[],
             datas=[('CDSAXS_gui_ui.ui', '.'), ('misc_settings_ui.ui', '.'), ('slice_center_pitch_ui.ui', '.'),
             ('../readme_processing.md.html', '.'), ('../markdeep_qwebview', '.'), ('../images', 'images')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=['PyQt4'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='mainwindow',
          debug=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='mainwindow')
