import optparse
import json
import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw



'''
Description:
    This script helps visualize the results from the line detection engine.

Commands of usage:
    To Draw BaseLines
python arguments.py -b 1 -i image.jpg -o output.jpg -f lines.json

    To Draw Lines
python arguments.py -b 1 -i image.jpg -o output.jpg -f lines.json

    To Draw Both
python arguments.py --lb 1 -i image.jpg -o output.jpg -f lines.json


'''


parser = optparse.OptionParser()




# Add Boundries on the image
parser.add_option('-b', '--boundry',
    action="store", dest="boundry",
    help="Draw boundraies on the image", default="0")

# Add lines on the image
parser.add_option('-l', '--lines',
    action="store", dest="lines",
    help="Draw lines on the image", default="0")

# Add both lines and boundries on the image
parser.add_option('-t', '--bl',
    action="store", dest="both",
    help="# Add both lines and boundries on the image", default="0")

# Add both lines and boundries on the image
parser.add_option('-y', '--lb',
    action="store", dest="both1",
    help="# Add both lines and boundries on the image", default="0")


# Specify Image Path
parser.add_option('-i', '--image_path',
    action="store", dest="image_path",
    help="Specify Image Path Here", default="image.jpg")

# Specify save output path here
parser.add_option('-o', '--output',
    action="store", dest="save_path",
    help="# Specify save output path here", default="output.jpg")

# Add json file Path
parser.add_option('-f', '--file',
    action="store", dest="file",
    help="Add json file path here", default="lines.json")


options, args = parser.parse_args()
print(args, options)
print(type(options))
options = vars(options)
# Opening JSON file
with open(options["file"], 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)
    
# Reading Image    
image = Image.open(options["image_path"])




 

def draw_lines(key, image):
    '''
    Description:
        Draw Lines on the Image.
    
    Input:
        key    (str)  : Flag line or baseline respectively
        image  (PIL)  : Image on which you want to draw lines
    
    Output:
        image (PIL Image) : PIL image with drawn lines
    
    '''
    global json_object
    
    draw = ImageDraw.Draw(image) 
    if key == 'baseline':
        lines = json_object['lines']
        for i in lines:
            baseline = tuple([j for num in i['baseline'] for j in num])
            draw.line(line, fill=128, width=9)
    
    elif key == 'line':
        lines = json_object['lines']
        for i in lines:
            boundry = i['boundary']
            boundry.append(boundry[-1])
            boundry.append(boundry[0])
            line = [num for line in boundry for num in line]
            draw.line(line, fill=128, width=3)

    else:
        lines = json_object['lines']
        for i in lines:
            baseline = tuple([j for num in i['baseline'] for j in num])
            draw.line(baseline, fill=128)
            boundry = i['boundary']
            boundry.append(boundry[-1])
            boundry.append(boundry[0])
            line = [num for line in boundry for num in line]
            draw.line(line, fill=128, width=3)
            
    
    return image


key = "both"
if options["boundry"] != "0":
    key = "baseline"

if options["lines"] != "0":
    key = "line"


image = draw_lines(key, image)
image.save(options["save_path"])
