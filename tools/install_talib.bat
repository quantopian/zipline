powershell -Command "(New-Object Net.WebClient).DownloadFile('http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-msvc.zip', 'ta-lib-0.4.0-msvc.zip')"
powershell -Command "Add-Type -AssemblyName System.IO.Compression.FileSystem;[System.IO.Compression.ZipFile]::ExtractToDirectory('ta-lib-0.4.0-msvc.zip', 'C:\')"
pushd C:\ta-lib\c\
pushd make\cdr\win32\msvc
nmake
popd
popd
del ta-lib-0.4.0-msvc.zip
