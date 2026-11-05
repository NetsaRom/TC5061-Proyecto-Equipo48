## Instalación de WSL 2 y Podman en Windows

Para ejecutar la API dentro de un contenedor en Windows es necesario instalar Podman Desktop y configurar WSL 2, que proporciona el entorno Linux requerido para ejecutar los contenedores.

### 1. Habilitar WSL 2

**Requisitos:**
- Windows 10 compatible con WSL 2 o Windows 11.
- Virtualización habilitada en BIOS/UEFI.
- Permisos de administrador.

Abrir PowerShell como administrador y ejecutar:

```powershell
wsl --install
```

Este comando instala WSL y una distribución Linux predeterminada cuando sea necesario.

Si WSL ya está instalado, actualizarlo:

```powershell
wsl --update
```

Configurar WSL 2 como versión predeterminada:

```powershell
wsl --set-default-version 2
```

Verificar la instalación:

```powershell
wsl --status
```

Consultar las distribuciones instaladas:

```powershell
wsl --list --verbose
```

Las distribuciones deben utilizar la versión 2 de WSL.

**Nota:** si Windows solicita reiniciar el equipo, realizar el reinicio antes de continuar.

#### Si WSL no se instala mediante el comando anterior

En versiones de Windows donde sea necesario habilitar manualmente las características, abrir PowerShell como administrador y ejecutar:

```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
```

```powershell
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

Después, reiniciar Windows y ejecutar nuevamente:

```powershell
wsl --update
```

### 2. Instalar Podman Desktop

Descargar Podman Desktop desde su página oficial:

https://podman-desktop.io/downloads

Pasos de instalación:

1. Descargar el instalador correspondiente a Windows.
2. Ejecutar el instalador.
3. Seguir las instrucciones de instalación.
4. Abrir Podman Desktop.
5. Configurar Podman seleccionando WSL 2 como proveedor de virtualización, cuando se solicite.
6. Finalizar la configuración de la máquina de Podman.

### 3. Configurar la máquina de Podman

Abrir PowerShell y verificar que Podman esté instalado:

```powershell
podman --version
```

Consultar las máquinas disponibles:

```powershell
podman machine list
```

Si todavía no existe una máquina de Podman, crearla:

```powershell
podman machine init
```

Iniciar la máquina:

```powershell
podman machine start
```

Verificar que se encuentre en ejecución:

```powershell
podman machine list
```

El estado de la máquina debe aparecer como `Running`.

**Nota:** si Podman Desktop ya creó una máquina durante su configuración, no es necesario ejecutar nuevamente `podman machine init`. Solamente hay que iniciarla si está detenida.

### 4. Verificar el funcionamiento de Podman

Para comprobar que Podman puede ejecutar contenedores, utilizar:

```powershell
podman run --rm hello-world
```

Si la instalación es correcta, se mostrará un mensaje de bienvenida de Podman.

También se puede verificar la versión instalada mediante:

```powershell
podman --version
```

Con estos pasos, Windows queda preparado para construir y ejecutar la imagen de FastAPI correspondiente al proyecto MiBici.

### 5. Continuar con el despliegue del proyecto

Una vez instalado y configurado Podman, continuar con la sección **Construcción de la imagen con Podman** de este documento.

La instalación de WSL 2 y Podman se realiza una sola vez por equipo. Posteriormente, solamente será necesario iniciar la máquina de Podman para trabajar con los contenedores.
