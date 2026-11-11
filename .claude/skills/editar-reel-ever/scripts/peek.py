import sys, subprocess, os
from PIL import Image, ImageDraw
d=sys.argv[1]; ts=[int(x) for x in sys.argv[2:]]; os.makedirs(d,exist_ok=True)
cols=4; rows=(len(ts)+3)//4; W=Image.new('RGB',(320*cols,180*rows))
for i,t in enumerate(ts):
    f=f"{d}/p{t}.jpg"
    subprocess.run(["ffmpeg","-v","error","-y","-ss",str(t),"-i","../dl/video720.mp4","-frames:v","1","-vf","scale=320:180",f])
    im=Image.open(f); ImageDraw.Draw(im).text((5,5),f"{t//60}:{t%60:02d}",fill='yellow'); W.paste(im,(320*(i%cols),180*(i//cols)))
W.save(f"{d}/g.jpg")
