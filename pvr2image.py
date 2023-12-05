'''
MIT License

Copyright (c) 2023 VincentNL

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

from PIL import Image,ImagePalette
import os
import sys
import math
import tkinter as tk
from tkinter import filedialog

debug = False

def read_col(px_format, color):

    if px_format==0: # ARGB1555
        a = 0xff if ((color >> 15) & 1) else 0
        r = (color >> (10 - 3)) & 0xf8
        g = (color >> (5 - 3)) & 0xf8
        b = (color << 3) & 0xf8
        return (r, g, b, a)

    elif px_format==1: # RGB565
        a = 0xff
        r = (color >> (11 - 3)) & (0x1f << 3)
        g = (color >> (5 - 2)) & (0x3f << 2)
        b = (color << 3) & (0x1f << 3)
        return (r, g, b, a)

    elif px_format==2:  # ARGB4444
        a = (color >> (12 - 4)) & 0xf0
        r = (color >> (8 - 4)) & 0xf0
        g = (color >> (4 - 4)) & 0xf0
        b = (color << 4) & 0xf0
        return (r, g, b, a)

    elif px_format==5: # RGB555
        a = 0xFF
        r = (color >> (10 - 3)) & 0xf8
        g = (color >> (5 - 3)) & 0xf8
        b = (color << 3) & 0xf8
        return (r, g, b, a)

    elif px_format in [7]: # ARGB8888
        a = (color >> 24) & 0xFF
        r = (color >> 16) & 0xFF
        g = (color >> 8) & 0xFF
        b = (color >> 0) & 0xFF
        return (r, g, b, a)

    elif px_format in [14]: # RGBA8888
        r = (color >> 24) & 0xFF
        g = (color >> 16) & 0xFF
        b = (color >> 8) & 0xFF
        a = (color >> 0) & 0xFF
        return (r, g, b, a)

    elif px_format==3:

        # YUV422
        yuv0, yuv1 = color

        y0 = (yuv0 >> 8) & 0xFF
        u = yuv0 & 0xFF
        y1 = (yuv1 >> 8) & 0xFF
        v = yuv1 & 0xFF

        # Perform YUV to RGB conversion
        c0 = y0 - 16
        c1 = y1 - 16
        d = u - 128
        e = v - 128

        r0 = max(0, min(255, int((298 * c0 + 409 * e + 128) >> 8)))
        g0 = max(0, min(255, int((298 * c0 - 100 * d - 208 * e + 128) >> 8)))
        b0 = max(0, min(255, int((298 * c0 + 516 * d + 128) >> 8)))

        r1 = max(0, min(255, int((298 * c1 + 409 * e + 128) >> 8)))
        g1 = max(0, min(255, int((298 * c1 - 100 * d - 208 * e + 128) >> 8)))
        b1 = max(0, min(255, int((298 * c1 + 516 * d + 128) >> 8)))

        return r0, g0, b0, r1, g1, b1



def read_pal(mode, color, act_buffer):

    if mode == 4444:
        red = ((color >> 8) & 0xf) << 4
        green = ((color >> 4) & 0xf) << 4
        blue = (color & 0xf) << 4
        alpha = '-'

    if mode == 555:
        red = ((color >> 10) & 0x1f) << 3
        green = ((color >> 5) & 0x1f) << 3
        blue = (color & 0x1f) << 3
        alpha = '-'

    elif mode == 565:
        red = ((color >> 11) & 0x1f) << 3
        green = ((color >> 5) & 0x3f) << 2
        blue = (color & 0x1f) << 3
        alpha = '-'

    elif mode == 8888:
        blue = (color >> 0) & 0xFF
        green = (color >> 8) & 0xFF
        red = (color >> 16) & 0xFF
        alpha = (color >> 24) & 0xFF

    act_buffer += bytes([red, green, blue])

    if debug:
        bits = [(color >> i) & 1 for i in range(16)]

        if mode == 4444:
            d_blue = f"{bits[0:4]}"
            d_green = f"{bits[4:8]}"
            d_red = f"{bits[8:12]}"

        if mode == 565:
            d_blue = f"{bits[0:5]}"
            d_green = f"{bits[5:11]}"
            d_red = f"{bits[11:16]}"

        elif mode == 555:
            d_blue = f"{bits[0:5]}"
            d_green = f"{bits[5:10]}"
            d_red = f"{bits[11:16]}"

        elif mode == 8888:
            d_blue = ""
            d_green = ""
            d_red = ""

        print(
            f"Palette {int(len(act_buffer)/3)} Blue:{blue} {d_blue},Green:{green} {d_green},Red:{red} {d_red},Alpha:{alpha}")
    return act_buffer

def read_pvp(f):
    global act_buffer

    f.seek(0x08)
    pixel_type = int.from_bytes(f.read(1), 'little')
    if pixel_type == 1:
        mode = 565
    elif pixel_type == 2:
        mode = 4444
    elif pixel_type == 6:
        mode = 8888
    else:
        mode = 555
    if debug: print(mode)

    f.seek(0x0e)
    ttl_entries = int.from_bytes(f.read(2), 'little')

    f.seek(0x10)  # Start palette data
    current_offset = 0x10
    act_buffer = bytearray()

    for counter in range(0, ttl_entries):
        if mode != 8888:
            color = int.from_bytes(f.read(2), 'little')
            act_buffer = read_pal(mode, color, act_buffer)
            current_offset += 0x2
        else:
            color = int.from_bytes(f.read(4), 'little')
            act_buffer = read_pal(mode, color, act_buffer)
            current_offset += 0x4

    return act_buffer, mode, ttl_entries

def write_png(img,bits):
    if not os.path.exists(ext_dir + '\Extracted'):
        os.makedirs(ext_dir + '\Extracted')
    img.save((ext_dir + '\Extracted/' + file_name[:-4] + ".png"), bits=bits)

def write_act(act_buffer):

    if not os.path.exists(ext_dir + '\Extracted\ACT'):
        os.makedirs(ext_dir + '\Extracted\ACT')

    with open((ext_dir + '\Extracted\ACT/' + file_name[:-4] + ".ACT"), 'w+b') as n:
        if debug:(ext_dir + '\Extracted\ACT/' + file_name[:-4] + ".ACT")

        # Pad file with 0x00 if 16-color palette

        if len(act_buffer) < 768:
            act_file = bytes(act_buffer) + bytes(b'\x00' * (768 - len(act_buffer)))
        else:
            act_file = bytes(act_buffer)
        n.write(act_file)

def detwiddle(w, h):
    # Initialize variables
    index = 0
    pat2, h_inc, arr, h_arr = [], [], [], []

    # Build Twiddle index table
    seq = [2, 6, 2, 22, 2, 6, 2]
    pat = seq + [86] + seq + [342] + seq + [86] + seq

    for i in range(4):
        pat2 += [1366, 5462, 1366, 21846]
        pat2 += [1366, 5462, 1366, 87382] if i % 2 == 0 else [1366, 5462, 1366, 349526]

    for i in range(len(pat2)):
        h_inc.extend(pat + [pat2[i]])
    h_inc.extend(pat)

    # Rectangle (horizontal)
    if w > h:
        ratio = int(w/h)
        if debug:
            print(f'width is {ratio} times height!')

        if w % 32 == 0 and w & (w - 1) != 0 or h & (h - 1) != 0:
            if debug:
                print('h and w not power of 2. Using Stride format')
            n = h * w
            for i in range(n):
                arr.append(i)
        else:
            # Single block h_inc length
            cur_h_inc = {w: h_inc[0:h - 1] + [2]}  # use height size to define repeating block h_inc

            # define the first horizontal row of image pixel array:
            for j in range(ratio):
                if w in cur_h_inc:
                    for i in cur_h_inc[w]:
                        h_arr.append(index)
                        index += i
                index = (len(h_arr) * h)

            # define the vertical row of image pixel array of repeating block:
            v_arr = [int(x / 2) for x in h_arr]
            v_arr = v_arr[0:h]

            for val in v_arr:
                arr.extend([x + val for x in h_arr])

    # Rectangle (vertical)
    elif h > w:
        ratio = int(h/w)
        if debug:
            print(f'height is {ratio} times width!')

        # Set the size of pixel increase array
        cur_h_inc = {w: h_inc[0:w - 1] + [2]}

        # define the first horizontal row of image pixel array:
        if w in cur_h_inc:
            for i in cur_h_inc[w]:
                h_arr.append(index)
                index += i

        # define the vertical row of image pixel array:
        v_arr = [int(x / 2) for x in h_arr]

        # Repeat vertical array block from the last value of array * h/w ratio
        for i in range(ratio):
            if i == 0:
                last_val = 0
            else:
                last_val = arr[-1] + 1

            for val in v_arr:
                arr.extend([last_val + x + val for x in h_arr])

    elif w == h:  # Square
        cur_h_inc = {w: h_inc[0:w - 1] + [2]}
        # define the first horizontal row of image pixel array:
        if w in cur_h_inc:
            for i in cur_h_inc[w]:
                h_arr.append(index)
                index += i

        # define the vertical row of image pixel array:
        v_arr = [int(x / 2) for x in h_arr]

        for val in v_arr:
            arr.extend([x + val for x in h_arr])

    return arr


def decode_pvr(f,w, h, offset=None,px_format=None,tex_format=None):

    # open the file a.pvr and read every byte according to the list

    f.seek(offset)
    data = bytearray()

    # Exclude non twiddled formats + VQ
    if tex_format not in [9,10,11,12,14,15]:
        arr = detwiddle(w, h)

    # Palettized images loop
    if tex_format in [5,6,7,8]:
        if tex_format in [7, 8]:  # 8bpp
            palette_entries = 256
            pixels = bytes(f.read(w * h))  # read only required amount of bytes

        else:  # 4bpp , convert to 8bpp

            palette_entries = 16
            pixels = bytes(f.read(w * h // 2))  # read only required amount of bytes
            for i in range(len(pixels)):
                data.append(((pixels[i]) & 0x0f) * 0x11)  # last 4 bits
                data.append((((pixels[i]) & 0xf0) >> 4) * 0x11)  # first 4 bits

            pixels = data  # converted 4bpp --> 8bpp "data" back into "pixels" variable

        data = bytearray()
        for num in arr:
            data.append(pixels[num])

        # create a new image with grayscale data
        img = Image.new('L', (w, h))
        img.putdata(bytes(data))

        if apply_palette == 1:
            new_palette = ImagePalette.raw("RGB", bytes(act_buffer))
        else:
            new_palette = ''

        if palette_entries == 16:

            img = img.convert('RGB')
            img = img.convert('L', colors=16)

            # 16-col greyscale palette
            pal_16_grey = []
            for i in range(0, 16):
                pal_16_grey += [i * 17, i * 17, i * 17]

            # print(palette)
            img = img.convert('P', colors=16)
            img.putpalette(pal_16_grey)

            if new_palette != '':
                # Set the image's palette to the loaded palette

                # Get the palette from the image
                img.getpalette()
                img.putpalette(new_palette)

            write_png(img,4)

            # save the image
            img.save((ext_dir + '\Extracted/' + file_name[:-4] + ".png"), bits=4)

        else:
            # Convert the image to a palettized grayscale 256 col
            img = img.convert('L', colors=256)
            img = img.convert('P', colors=256)

            # Get the palette from the image
            palette = img.getpalette()

            # Set the palette in the same order as before
            img.putpalette(palette)
            if new_palette != '':
                # Set the image's palette to the loaded palette
                img.putpalette(new_palette)

            # save the image
            write_png(img,8)

    # VQ
    elif tex_format in [3,4,16,17]:

        codebook_size = 256

        # SmallVQ - Thanks Kion! :)

        if tex_format == 16:
            if w <= 16:
                codebook_size = 16
            elif w == 32:
                codebook_size = 32
            elif w == 64:
                codebook_size = 128
            else:
                codebook_size = 256

        # SmallVQ + Mips
        elif tex_format == 17:
            if w <= 16:
                codebook_size = 16
            elif w == 32:
                codebook_size = 64
            else:
                codebook_size = 256

        # print(codebook_size)

        codebook = []

        if px_format not in [3]:
            for l in range(codebook_size):
                block = []
                for i in range(4):
                    pixel = (int.from_bytes(f.read(2), 'little'))
                    pix_col = read_col(px_format, pixel)
                    block.append(pix_col)

                codebook.append(block)

        # YUV422

        else:

            yuv_codebook = []
            for l in range(codebook_size):
                block = []
                for i in range(4):
                    pixel = (int.from_bytes(f.read(2), 'little'))
                    block.append(pixel)

                r0, g0, b0, r1, g1, b1 = read_col(px_format, (block[0], block[3]))
                r2, g2, b2, r3, g3, b3 = read_col(px_format, (block[1], block[2]))

                yuv_codebook.append([(r0, g0, b0), (r2, g2, b2), (r3, g3, b3), (r1, g1, b1)])

            codebook = yuv_codebook

        #print('codebook entries:', len(codebook))


        #print('offset:', hex(f.tell()))

        # VQ Mips!
        if tex_format in [4,17]:

            pvr_dim = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
            mip_size = [0x10, 0x40, 0x100, 0x400, 0x1000, 0x4000, 0x10000, 0x40000]
            size_adjust = {4: 1, 17: 1}  # 8bpp size is 4bpp *2
            extra_mip = {4: 0x6, 17: 0x6,}  # smallest mips fixed size

            for i in range(len(pvr_dim)):
                if pvr_dim[i] == w:
                    mip_index = i - 1
                    break

            # Skip mips for image data offset
            mip_sum = (sum(mip_size[:mip_index]) * size_adjust[tex_format]) + (extra_mip[tex_format])

            f.seek(f.tell() + mip_sum)
            #print(hex(f.tell()))

        # Read pixel_index:
        pixel_list = []
        bytes_to_read = int((w*h)/4)

        # Each index stores 4 pixels
        for i in range(bytes_to_read):
            pixel_index = (int.from_bytes(f.read(1), 'little'))
            pixel_list.append(int(pixel_index))

        # Detwiddle image data indices, put them into arr list
        arr = detwiddle(int(w / 2), int(h / 2))
        img = Image.new('RGBA', (w, h))

        # i represent the current index from arr list
        i = 0

        for y in range(int(h / 2)):
            for x in range(int(w / 2)):
                img.putpixel((x * 2 + 0, y * 2 + 0), codebook[pixel_list[arr[i]]][0])
                img.putpixel((x * 2 + 1, y * 2 + 0), codebook[pixel_list[arr[i]]][2])
                img.putpixel((x * 2 + 0, y * 2 + 1), codebook[pixel_list[arr[i]]][1])
                img.putpixel((x * 2 + 1, y * 2 + 1), codebook[pixel_list[arr[i]]][3])
                i += 1

        # save the image
        write_png(img,'')

    # BMP ABGR8888
    elif tex_format in [14,15]:
        pixels = [int.from_bytes(f.read(4), 'little') for _ in range(w * h)]
        rgb_values = [(read_col(14, p)) for p in pixels]

        # Create a new image and set pixel values using putdata
        img = Image.new('RGBA', (w, h))
        img.putdata(rgb_values)

        # save the image
        write_png(img,'')

    # BUMP loop
    elif px_format == 4:
        pixels = [int.from_bytes(f.read(2), 'little') for _ in range(w * h)]
        rgb_values = [cart_to_rgb(process_SR(p)) for p in (pixels[i] for i in arr)]

        # Create a new image and set pixel values using putdata
        img = Image.new('RGB', (w, h))
        img.putdata(rgb_values)

        # save the image
        write_png(img,'')



    # ARGB modes
    elif px_format in [0,1,2,5,7,18]:

        pixels = [int.from_bytes(f.read(2), 'little') for _ in range(w * h)]

        if tex_format not in [9, 10, 11, 12, 14, 15]:  # If Twiddled
            rgb_values = [(read_col(px_format, p)) for p in (pixels[i] for i in arr)]
        else:
            rgb_values = [(read_col(px_format, p)) for p in pixels]

        # Create a new image and set pixel values using putdata
        img = Image.new('RGBA', (w, h))
        img.putdata(rgb_values)

        # save the image
        write_png(img,'')

    # YUV420 modes
    elif px_format in [6]:
        rgb_values=[]
        yuv420_to_rgb(f,w,h,rgb_values)

        img = Image.new('RGB', (w, h))
        img.putdata(rgb_values)

        # save the image
        write_png(img, '')


    # YUV422 modes
    elif px_format in [3]:
        rgb_values = []
        img = Image.new('RGB', (w, h))

        # Twiddled
        if tex_format not in [9, 10, 11, 12, 14, 15]:
            i = 0
            offset = f.tell()

            for y in range(h):
                for x in range(0, w, 2):
                    f.seek(offset+(arr[i]*2))
                    yuv0 = int.from_bytes(f.read(2), 'little')
                    i += 1
                    f.seek(offset + (arr[i] * 2))
                    yuv1 = int.from_bytes(f.read(2), 'little')
                    r0, g0, b0, r1, g1, b1 = read_col(px_format, (yuv0, yuv1))
                    rgb_values.append((r0, g0, b0))
                    rgb_values.append((r1, g1, b1))
                    i += 1


        else:
            for y in range(h):
                for x in range(0, w, 2):
                    # Read yuv0 and yuv1 separately
                    yuv0 = int.from_bytes(f.read(2), 'little')
                    yuv1 = int.from_bytes(f.read(2), 'little')
                    r0, g0, b0, r1, g1, b1 = read_col(px_format, (yuv0, yuv1))
                    rgb_values.append((r0, g0, b0))
                    rgb_values.append((r1, g1, b1))

        img.putdata(rgb_values)
        # save the image
        write_png(img, '')


def load_pvr(PVR_file):

    try:
        with open(PVR_file, 'rb') as f:
            header_data = f.read()
            offset = header_data.find(b"PVRT")
            if offset != -1 or len(header_data) < 0x10:
                if debug: print("Position of 'PVRT' text:", hex(offset))
                f.seek(offset + 0x8)

                # Pixel format
                px_format = int.from_bytes(f.read(1), byteorder='little')
                tex_format = int.from_bytes(f.read(1), byteorder='little')

                f.seek(f.tell()+2)

                # Image size
                w = int.from_bytes(f.read(2), byteorder='little')
                h = int.from_bytes(f.read(2), byteorder='little')
                offset = f.tell()

                if debug: print(PVR_file.split('/')[-1],'size:', w, 'x', h, 'format:', f'[{tex_format}] {tex_modes[tex_format]}',f'[{px_format}] {px_modes[px_format]}')

                if tex_format not in [5,6,7,8] or h > 1024 or w > 1024:
                    if debug: print(PVR_file, "Not Palettized PVR!")

                if tex_format in [2,4,6,8,10,12,15,17,18]:
                    #print('mip-maps!')

                    if tex_format in [2,6,8,10,15,18]:
                        # Mips skip
                        pvr_dim = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
                        mip_size = [0x20, 0x80, 0x200, 0x800, 0x2000, 0x8000, 0x20000, 0x80000]
                        size_adjust = {2: 4, 6: 1, 8: 2,10:4,15:8,18:4}  # 8bpp size is 4bpp *2
                        extra_mip = {2: 0x2c ,6: 0xc, 8: 0x17,10:0x2c,15:0x54,18: 0x30}  # smallest mips fixed size

                        for i in range(len(pvr_dim)):
                            if pvr_dim[i] == w:
                                mip_index = i - 1
                                break

                        # Skip mips for image data offset
                        mip_sum = (sum(mip_size[:mip_index]) * size_adjust[tex_format]) + (extra_mip[tex_format])

                        offset += mip_sum
                        #print(hex(offset))

                decode_pvr(f, w, h, offset, px_format, tex_format)

            else:
                if debug: print("'PVRT' header not found!")

    except: print(f'PVR data error! {PVR_file}')


def load_pvp(PVP_file):

    try:
        with open(PVP_file, 'rb') as f:
            file_size = len(f.read())
            f.seek(0x0)
            PVP_check = f.read(4)

            if PVP_check == b'PVPL' and file_size > 0x10:  # PVPL header and size is OK!
                act_buffer, mode, ttl_entries = read_pvp(f)
                write_act(act_buffer)
            else:
                print('Invalid .PVP file!')  # Skip this file
    except:
         print(f'PVP data error! {PVP_file}')

def process_SR(SR_value):
    S = (1.0 - ((SR_value >> 8) / 255.0)) * math.pi / 2
    R = (SR_value & 0xFF) / 255.0 * 2 * math.pi - 2 * math.pi * (SR_value & 0xFF > math.pi)
    red = (math.sin(S) * math.cos(R) + 1.0) * 0.5
    green = (math.sin(S) * math.sin(R) + 1.0) * 0.5
    blue = (math.cos(S) + 1.0) * 0.5
    return red, green, blue

def cart_to_rgb(cval):
    return tuple(int(c * 255) for c in cval)

def yuv420_to_rgb(f,w,h,rgb_values):
    # Credits to Egregiousguy for YUV420 --> YUV420P conversion
    buffer = bytearray()

    col = w // 16
    row = h // 16

    U = [bytearray() for _ in range(8 * row)]
    V = [bytearray() for _ in range(8 * row)]
    Y01 = [bytearray() for _ in range(col)]
    Y23 = [bytearray() for _ in range(col)]

    for i in range(1, row + 1):
        for n in range(8):
            U[n + 8 * (i - 1)] = bytearray()

        for n in range(col):
            Y01[n] = bytearray()
            Y23[n] = bytearray()

        for _ in range(col):
            for n in range(8):
                U[n + 8 * (i - 1)] += f.read(0x8)

            for n in range(8):
                V[n + 8 * (i - 1)] += f.read(0x8)

            for _ in range(2):
                for n in range(8):
                    Y01[n] += f.read(0x8)

            for _ in range(2):
                for n in range(8):
                    Y23[n] += f.read(0x8)

        for n in range(col):
            buffer += Y01[n]

        for n in range(col):
            buffer += Y23[n]

    for data in U + V:
        buffer += data

    # Extract Y, U, and V components from the buffer
    Y = list(buffer[:int(w * h)])
    U = list(buffer[int(w * h):int(w * h * 1.25)])
    V = list(buffer[int(w * h * 1.25):])

    # Reshape Y, U, and V components
    Y = [Y[i:i + w] for i in range(0, len(Y), w)]
    U = [U[i:i + w // 2] for i in range(0, len(U), w // 2)]
    V = [V[i:i + w // 2] for i in range(0, len(V), w // 2)]

    # Upsample U and V components
    U = [item for sublist in U for item in [item for item in sublist] * 2]
    V = [item for sublist in V for item in [item for item in sublist] * 2]

    # Reshape U and V components after upsampling
    U = [U[i:i + w] for i in range(0, len(U), w)]
    V = [V[i:i + w] for i in range(0, len(V), w)]

    # Convert YUV to RGB
    for i in range(h):
        for j in range(w):
            i_UV = min(i // 2, len(U) - 1)
            j_UV = min(j // 2, len(U[i_UV]) - 1)
            y, u, v = Y[i][j], U[i_UV][j_UV], V[i_UV][j_UV]
            r = int(max(0, min(255, round(y + 1.402 * (v - 128)))))
            g = int(max(0, min(255, round(y - 0.344136 * (u - 128) - 0.714136 * (v - 128)))))
            b = int(max(0, min(255, round(y + 1.772 * (u - 128)))))
            rgb_values.append((r, g, b))
    return rgb_values

def main():
    global ext_dir,file_name,apply_palette,px_modes,tex_modes

    px_modes = {
        0: 'ARGB1555',
        1: 'RGB565',
        2: 'ARGB4444',
        3: 'YUV422',
        4: 'BUMP',
        5: 'RGB555',
        6: 'YUV420',
        7: 'ARGB8888',
        8: 'PAL-4',
        9: 'PAL-8',
        10: 'AUTO',
    }

    # Tex format
    tex_modes = {
        1: 'Twiddled',
        2: 'Twiddled Mips',
        3: 'Twiddled VQ',
        4: 'Twiddled VQ Mips',
        5: 'Twiddled Pal4 (16-col)',
        6: 'Twiddled Pal4 + Mips (16-col)',
        7: 'Twiddled Pal8 (256-col)',
        8: 'Twiddled Pal8 + Mips (256-col)',
        9: 'Rectangle',
        10: 'Rectangle + Mips',
        11: 'Stride',
        12: 'Stride + Mips',
        13: 'Twiddled Rectangle',
        14: 'BMP',
        15: 'BMP + Mips',
        16: 'Twiddled SmallVQ',
        17: 'Twiddled SmallVQ + Mips',
        18: 'Twiddled Alias + Mips',
    }

    # Use Tkinter file selector to load .PVP files

    root = tk.Tk()
    root.withdraw()

    my_file = filedialog.askopenfilenames(
        initialdir=".",
        title="Select .PVR / .PVP files to convert",
        filetypes=[(".pvr .pvp files", ".PV*")]
    )

    # Use Tkinter file selector to save exported images
    if my_file != "":
        root = tk.Tk()
        root.withdraw()

        ext_dir = filedialog.askdirectory(
            initialdir=".",
            title="Select extraction folder",
        )

    # remove companion .PVP/.PVR, filter the list

    new_list = []
    for item in my_file:
        key = item[:-4]
        if not any(key == x[:-4] for x in new_list):
            new_list.append(item)
    my_file = new_list

    selected_files = len(my_file)
    current_file = 0

    # create Extracted\ACT folders
    if debug:print(ext_dir + '\Extracted\ACT')

    # -----------
    # Loop start
    # -----------

    while current_file < selected_files:  # Process all selected files in the list
        if my_file:  # if at least one file selected
            cur_file = my_file[current_file]
            dir_path, file_name = os.path.split(cur_file)
            filetype = cur_file[-4:]
            PVR_file = cur_file[:-4] + '.pvr'  # If .PVR file exists in the same folder
            PVP_file = cur_file[:-4] + '.pvp'  # If .PVP file exists in the same folder

            if debug:print('filetype:',filetype)

            # If cur_file is a .PVP
            if filetype == ".pvp" or filetype == ".PVP":
                load_pvp(PVP_file)

                if os.path.exists(PVR_file):
                    apply_palette = 1
                    load_pvr(PVR_file)

                    if debug:print('companion PVR exist!')

                else:
                    apply_palette = 0
                    if debug:print('Only convert PVP to ACT!')


            # If cur_file is a .PVR
            elif filetype == ".pvr" or filetype == ".PVR":

                if os.path.exists(PVP_file):
                    if debug:print('companion PVP exist!')
                    apply_palette = 1
                    load_pvp(PVP_file)

                else:
                    apply_palette = 0
                    if debug:print('Only convert PVR to grayscale .png!')

                load_pvr(PVR_file)

            current_file += 1

            if debug:print(cur_file)

main()
