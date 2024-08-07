# pvr2image

Convert **DC / Naomi** `.PVR` files to `.png`, `.tga`, or `.bmp`

You are free to use my code for any of your projects, just be kind and give due credits :)


If you like my work and want to support or just show some love:

https://ko-fi.com/vincentnl

https://www.patreon.com/VincentNL

# How to use

- [Download](https://github.com/VincentNLOBJ/pvr2image/releases) latest Win release or get `pvr2image.py` 
- Open `pvr2image.exe` or `pvr2image.py`, select .pvr or .pvp (palette) files.
- Select extract folder
- Extracted files will be placed in a subfolder called `Extracted`, palettes in `ACT` folder.

Please note:
- If `.pvr` and companion `.pvp` are in the same folder with same name, the image will be exported with palette.
- `.pvp` are automatically converted to `.act` for use with Photoshop or Gimp.
- `.png` is the default export image type, you can change that by altering `pvr2image.py` code

# Credits
* Egregiousguy for YUV420 to YUV420p conversion
* Kion for VQ handling and logic
* tvspelsfreak for SR conversion info on Bump to normal map
* MetalliC and Tcaw for 4444/565/555/1555 modes color value fix

# Supported PVR types

Texture Format:
* ARGB_1555
* RGB_565
* ARGB_4444
* YUV_422
* BUMP
* RGB_555
* ARGB_8888	
* YUV_420

Pixel Format:
* TWIDDLED
* TWIDDLED_MM
* VQ
* VQ_MM
* PALETTIZE4
* PALETTIZE4_MM
* PALETTIZE8
* PALETTIZE8_MM
* RECTANGLE
* STRIDE
* TWIDDLED_RECTANGLE
* ABGR
* ABGR_MM
* SMALLVQ
* SMALLVQ_MM
* TWIDDLED_MM_DMA
