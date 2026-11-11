import sys, subprocess
from PIL import Image, ImageDraw
f=sys.argv[1]; ts=[float(x) for x in sys.argv[2:]]
W=Image.new('RGB',(270*len(ts),480))
for i,t in enumerate(ts):
    subprocess.run(["ffmpeg","-v","error","-y","-ss",str(t),"-i",f,"-frames:v","1","-vf","scale=270:480","_sheet_tmp.jpg"])
    im=Image.open("_sheet_tmp.jpg"); ImageDraw.Draw(im).text((4,4),str(t),fill="lime"); W.paste(im,(270*i,0))
W.save(sys.argv[1]+".sheet.jpg")
