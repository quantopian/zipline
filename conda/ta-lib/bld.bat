rmdir /s /q "C:\ta-lib\"
powershell -Command "(New-Object Net.WebClient).DownloadFile('http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-msvc.zip', 'ta-lib-0.4.0-msvc.zip')"
IF %ERRORLEVEL% == 1; exit 1
powershell -Command "Add-Type -AssemblyName System.IO.Compression.FileSystem;[System.IO.Compression.ZipFile]::ExtractToDirectory('ta-lib-0.4.0-msvc.zip', 'C:\')"
IF %ERRORLEVEL% == 1; exit 1
pushd C:\ta-lib\c\
pushd make\cdd\win32\msvc
nmake
IF %ERRORLEVEL% == 1; exit 1
popd
pushd make\cdr\win32\msvc
nmake
IF %ERRORLEVEL% == 1; exit 1
popd
pushd make\cmd\win32\msvc
nmake
IF %ERRORLEVEL% == 1; exit 1
popd
pushd make\cmr\win32\msvc
nmake
IF %ERRORLEVEL% == 1; exit 1
popd
pushd make\csd\win32\msvc
nmake
IF %ERRORLEVEL% == 1; exit 1
popd
pushd make\csr\win32\msvc
nmake
IF %ERRORLEVEL% == 1; exit 1
popd
popd
del ta-lib-0.4.0-msvc.zip

python setup.py build --compiler msvc
python setup.py install --prefix=%PREFIX%
