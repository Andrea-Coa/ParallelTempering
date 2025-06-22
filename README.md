# Parallel Tempering

### Setup con Windows

1. Instalar Visual Studio, ya que usaremos el compilador MSVC.
2. Instalar Microsoft MPI ([link de referencia, que puede haber cambiado](https://www.microsoft.com/en-us/download/details.aspx?id=57467))
3. Ubicar las carpetas de ```\include``` y ```\lib``` de MS MPI SDKs (cambia según dónde hiciste la instalación). Por ejemplo:

```
    D:\Program Files\Microsoft SDKs\MPI\Include

	D:\Program Files\Microsoft SDKs\MPI\Lib\x64
```

Asimismo, ubicar el archivo ```vcvars64.bat```. Este sirve para activar el Visual Studio Developer Command Prompt. Por ejemplo, una ubicación podría ser:

```
D:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat
```

4. Crear el siguiente script para Powershell ```setup-msvc-mpi.ps1```. Al escribir los paths de ```vcvars64.bat```, así como de ```INCLUDE``` y ```LIB```, asegúrate de poner las rutas que identificaste en el paso 3:

```
# Run the MSVC environment setup
cmd /c '"D:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" && set' | ForEach-Object {
    if ($_ -match '^(.*?)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
    }
}

# Add MPI include and lib paths to environment (for this session)
$env:INCLUDE += ";D:\Program Files\Microsoft SDKs\MPI\Include"
$env:LIB += ";D:\Program Files\Microsoft SDKs\MPI\Lib\x64"

Write-Host "`n✅ MSVC and MPI environment ready!" -ForegroundColor Green

```

### Ejecutar el programa 

Una vez completados los pasos anteriores, para ejecutar un programa con MPI, por ejemplo ```ParallelTempering.cpp```, realizar lo siguiente:

1. En una sesión de Powershell, ejecutar:

```
. "setup-msvc-mpi.ps1"
```

Si tu script se encuentra en otro directorio, inclúyelo en la ruta a ```setup-msvc-mpi.ps1```. Por ejemplo, si lo tienes en ```D:/Scripts```, ejecuta ```. "D:\Scripts\setup-msvc-mpi.ps1"```

2. Compilar

```
cl ParallelTempering.cpp msmpi.lib
```

3. Ejecutar. Para especificar el número de procesos, usar ```-np```. Ejemplo con 4 procesos:

```
mpiexec -np 4 ParallelTempering.exe
```
