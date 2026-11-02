# Upscalers / Upscalers

`SOTC_UPSCALER`: unset (FSR 1), `fsr3`, `fsr4`, `ffxapi` ou / or `dlss`.
O lançador escolhe / the launcher picks: `FSR 1 (padrão)` / `FSR 3.1` /
`FSR 4.1.1 INT8 (build RDNA2)` / `FSR 3.1 via FFX API DLL (OptiScaler)`.

| Escolha / Choice | `SOTC_UPSCALER` | Notas / Notes |
|---|---|---|
| FSR 1 (padrão / default) | unset | Espacial, qualquer placa / spatial, any GPU |
| FSR 3.1 | `fsr3` | Temporal embutido / built-in temporal |
| FSR 4.1.1 INT8 (build RDNA2) | `fsr4` | Ativos locais, ver abaixo / local assets, see below |
| FSR 3.1 via FFX API DLL (OptiScaler) | `ffxapi` | Só Windows + Proton, sem fallback testado / Windows + Proton only, untested |
| NVIDIA DLSS (experimental) | `dlss` | RTX; FSR 1 nos menus e sem a placa / RTX; FSR 1 in menus and without the card |

Mais ajustes / further knobs:

| Opção / Option | Valor / Value |
|---|---|
| `SOTC_FSR4_DIR` | Pasta dos ativos FSR 4 / FSR 4 asset folder (padrão / default `<data>/fsr4_411`, onde `<data>` é `SOTC_DATA_DIR` ou a pasta do `sotc.py` / where `<data>` is `SOTC_DATA_DIR` or the folder holding `sotc.py`) |
| `SOTC_FFX_API_DLL` | Caminho da `amd_fidelityfx_vk.dll` / path of `amd_fidelityfx_vk.dll` (padrão / default: ao lado do executável no Windows, `libamd_fidelityfx_vk.so` no POSIX / next to the executable on Windows, `libamd_fidelityfx_vk.so` on POSIX) |
| `SOTC_FSR3_REACTIVE` | Máscara reativa do FSR 3 / FSR 3 reactive mask |
| `SOTC_UPSCALER_HUD` | HUD do upscaler / upscaler HUD |
| `SOTC_MIP_BIAS` | Viés de mip / mip bias (em `sampler.cpp`) |

## FSR 4

Os ativos são construídos uma vez da sua própria / the assets are built once from your own
`amd_fidelityfx_upscaler_dx12.dll` (4.1.1b INT8 RDNA2) + `amd_fidelityfx_loader_dx12.dll`:

```bash
bash tools/fsr4cap/build_assets.sh <upscaler> <loader>
```

Precisa de / needs `mingw-w64-gcc`, `umu-launcher`, um GE-Proton. Saída / output: `fsr4_411/`.
Nunca redistribuídos / never redistributed. Sem os ativos, `fsr4` volta ao / without the assets,
`fsr4` falls back to `fsr3`.

## OptiScaler

Só Windows sob Proton / Windows build under Proton only: `amd_fidelityfx_vk.dll` ao lado do /
next to `bb-probe.exe`, `SOTC_UPSCALER=ffxapi`, OptiScaler como / as `winmm.dll` com /
with `WINEDLSSOVERRIDES=winmm=n,b`. Sem o DLL, volta ao FSR 3.1 embutido / without the DLL,
falls back to built-in FSR 3.1. Não testado / untested.
