import os

local_path = os.path.dirname(os.path.realpath(__file__))

fps = 10
speed = 'fast' # ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
quality = 20 # range 20 - 40 ??
start_number = 30

infile = os.path.join(local_path, 'outputs2', 'muncie%03d.png')
outfile = os.path.join(local_path, 'outputs2', 'Muncie_RAS2D_depths.mp4')
cmd = f'ffmpeg -r {fps} -y -start_number {start_number} -i {infile} -vf "scale=710:804" -c:v libx264 -profile:v baseline -preset {speed} -tune animation -crf {quality} -pix_fmt yuv420p {outfile}'
print(cmd)
os.system(cmd)