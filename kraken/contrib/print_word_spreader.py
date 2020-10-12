#!/usr/bin/env python
#2020, Bruce Robertson
#Master file at https://github.com/brobertson/Lace2-tools/blob/master/normalize_hocr.py
import html, os, sys, argparse, PIL
from statistics import mean
from lxml import etree
from PIL import Image

#a custom exception to indiate when a page or other element doesn't 
#have a bounding box where we would expect it.
class BboxError(Exception):
    pass

#parse the arguments 
parser = argparse.ArgumentParser(description='''Convert kraken hocr output so
                                 that word bounding boxes are very likely to enclose the words, plus some space.
                                 This removes all spans of class ocrx_word that
                                 have single space text content. Its output is
                                 namespaced XHTML.''')
parser.add_argument('--inputDir', help='Path to directory where source files are found', required=True)
parser.add_argument('--outputDir', help='Path to directory where output is stored', required=True)
parser.add_argument('--imageDir', help='Path to directory where images corresponding to the html files are stored.', required=False)
parser.add_argument('-c', '--confidenceSummary', default=False, action="store_true", help="store summaries of word confidence in xhtml data- attributes and cut all material after the first ; from the word span title attribute, making their mouseover popups less obtrusive.")
parser.add_argument('-f', '--fixBigWordSpans', default=False, action="store_true", help="fix word_span elements whose bbox area is greater than a sixth of the whole page area by assigning them the bbox of the previous word.")
parser.add_argument('-s', '--shareSpaceSpans', default=False, action="store_true", help="normalize hocr output from kraken, which assigns a word to every space and gives it a bbox. This removes those space words and assigns their area to the words on either side, with some space in between, generating output more like Ocropus and tesseract.")
parser.add_argument("-v", "--verbose", help="increase output verbosity", default=False, action="store_true")
args = parser.parse_args()

def get_bbox_val(span, position):
    try:
        parts = html.unescape(span.get('title')).split(';')
        bbox_string = ""
        for part in parts:
            part = part.strip()
            if part.startswith('bbox'):
                bbox_string = part
        if bbox_string == "":
            if (args.verbose):
                print("couldn't find the bbox part!")
        return int(bbox_string.split(' ')[position+1])
    except Exception as e:
        #print("Exception getting title element on span {}".format(etree.tostring(span)))
        if (args.verbose):
            print(e)
            print("... therefore raising BboxError")
        raise BboxError

def get_bbox_area(span):
    try:
        width = get_bbox_val(span,2) - get_bbox_val(span,0)
        height = get_bbox_val(span,3) - get_bbox_val(span,1)
        area = width * height
        if (args.verbose):
            print("this element's area is " + str(area))
        return area
    except Exception as e:
        #print("Exception getting area on span  {}".format(etree.tostring(span)))
        raise

def set_bbox_value(span, position, val):
    try:
        parts = span.get('title').split(';')
    except Exception as e:
        print("Exception getting title element on span id {}.".format(span.get('id')))
        raise BboxError
    bbox_parts = parts[0].split(' ')
    bbox_parts[position + 1] = str(val)
    bbox_out = ' '.join(bbox_parts)
    parts[0] = bbox_out
    parts_out = ';'.join(parts)
    span.set('title', parts_out)

def share_space_spans(treeIn):
            right_max_fudge_factor = 7
            left_max_fudge_factor = 5
            space_spans = treeIn.xpath("//html:span[@class='ocrx_word'][text()=' ']",namespaces={'html':"http://www.w3.org/1999/xhtml"})
            #print('space spans: {}'.format(len(space_spans)))
            for space_span in space_spans:
                try:
                    previous_span = space_span.getprevious()
                except Exception as e:
                    print("Exception on parsing previous span with space id {}".format(space_span.get('id')))
                    print(e)
                    raise
                try:
                    next_span = space_span.getnext()
                except Exception as e:
                    print("Exception on parsing next span with space id {}".format(space_span.get('id')))
                    print(e)
                    raise
                #check that we have both
                if ((not previous_span is None) and (not next_span is None)):
                    #this means that there is both a previous and a next
                    if (args.verbose) :
                        print("***")
                        print("space_span title: {}".format(space_span.get('title')))
                        print("previous span title: {}".format(previous_span.get('title')))
                        print("next span title: {}".format(next_span.get('title')))
                    left_pos = get_bbox_val(previous_span,2)
                    right_pos = get_bbox_val(next_span,0)
                    middle =  int((left_pos + right_pos) / 2)
                    third = int((right_pos - left_pos) / 3)
                    left_fudge = min(third,left_max_fudge_factor)
                    right_fudge = min(third,right_max_fudge_factor)
                    if (args.verbose) :
                        print("left side: {0}; right side: {1}; middle: {2}".format(left_pos, right_pos, middle))
                    set_bbox_value(previous_span, 2, left_pos + left_fudge)
                    set_bbox_value(next_span, 0, right_pos - right_fudge)
                    if (args.verbose):
                        print(previous_span.text)
                        print("previous_span new title: {}".format(previous_span.get('title')))
                        print("next_span new title: {}".format(next_span.get('title')))
                #now remove the space span, no matter what
                space_span.getparent().remove(space_span)

def confidence_summary(treeIn):
    word_spans = treeIn.xpath("//html:span[@class='ocrx_word']",namespaces={'html':"http://www.w3.org/1999/xhtml"})
    for word_span in word_spans:
        try:
            #this gets the confidence values for each letter and represents them as a string list
            word_data = word_span.get('title').split(';')
            confs_string = word_data[1].split(' ')[2:]
            bbox_only = word_data[0]
            #convert to floats for math operations
            confs = test_list = [float(i) for i in confs_string]
            minimum = round(min(confs),2)
            average = round(mean(confs),2)
            #add attributes with these summary values
            word_span.set('data-min-confidence',str(minimum))
            word_span.set('data-average-confidence',str(average))
            word_span.set('title', bbox_only)
        except Exception as e:
            #there's not much to do if this goes wrong
            pass

def push_edge_spans_to_borders_of_line(treeIn):
    first_spans = treeIn.xpath("//html:span[@class='ocr_line']/html:span[@class='ocrx_word'][1]",namespaces={'html':"http://www.w3.org/1999/xhtml"})
    for span in first_spans:
        if (args.verbose):
            print("first span title: {}".format(span.get('title')))
        parent = span.getparent()
        line_l_edge = get_bbox_val(parent, 0)
        line_l_edge = line_l_edge + 1
        if (args.verbose):
            print("line_l_edge {}".format(line_l_edge))
        set_bbox_value(span, 0, line_l_edge)
    last_spans = treeIn.xpath("//html:span[@class='ocr_line']/html:span[@class='ocrx_word'][last()]",namespaces={'html':"http://www.w3.org/1999/xhtml"})
    for span in last_spans:
        parent = span.getparent()
        line_r_edge = get_bbox_val(parent,2)
        line_r_edge = line_r_edge - 1
        set_bbox_value(span,2,line_r_edge)

def fix_word_span_area(treeIn):
    word_spans = treeIn.xpath("//html:span[@class='ocrx_word'] | //html:span[@class='ocr_word']",namespaces={'html':"http://www.w3.org/1999/xhtml"})
    image_area = get_bbox_area(treeIn.xpath("//html:div[@class='ocr_page'][1]",namespaces={'html':"http://www.w3.org/1999/xhtml"})[0])
    #print("image area: {}".format(image_area))
    for span in word_spans:
        area = get_bbox_area(span)
        #print("word area:",area)
        if (area > image_area / 6):
            original_area = span.get('title')
            my_next = span.getnext()
            my_previous = span.getprevious()
            if (my_next):
                span.set('title',my_next.get('title'))
            elif (my_previous):
                span.set('title',my_previous.get('title'))
            else:
                span.set('title',span.getparent().get('title'))
            print("big word area, changing title attribute {}".format(original_area))
            print(etree.tostring(span))
            #span.getparent().remove(span)


def clean_ocr_page_title(xhtml, file_name):
   ocr_page = xhtml.xpath("//html:div[@class='ocr_page'][1]",namespaces={'html':"http://www.w3.org/1999/xhtml"})[0]
   #print(ocr_page)
   ocr_page_title = ocr_page.get('title')
   #print(ocr_page_title)
   sections = ocr_page_title.split(';')
   #print(sections)
   new_sections = "image " + (file_name.rsplit('.', 1)[0] + '.png') + "; " + sections[0]
   #print(new_sections)
   ocr_page.set('title',new_sections)
   return xhtml

def rewrite_ocr_page_title(xhtml, file_name, image_x, image_y):
    ocr_page = xhtml.xpath("//html:div[@class='ocr_page'][1]",namespaces={'html':"http://www.w3.org/1999/xhtml"})[0]
    image_file_name = (file_name.rsplit('.', 1)[0] + '.png')
    image_path = os.path.join(args.imageDir, image_file_name)
    image = Image.open(image_path)
    image_x, image_y = image.size
    new_title = "bbox 0 0 " + str(image_x) + " " + str(image_y) + ";image " + image_file_name
    ocr_page.set('title',new_title)
    return xhtml

if not(os.path.isdir(args.inputDir)):
    print('Input directory "'+image_dir+'" does not exist.\n\tExiting ...')
    sys.exit(1)

#Create the output directory if it doesn't exist
try:
    if not os.path.exists(args.outputDir):
        os.makedirs(args.outputDir, exist_ok=True)
except Exception as e:
    print("Error on creating output directory '" + args.outputDir +
    "':\n\t" + str(e) + "\n\tExiting ...")
    sys.exit(1)

if (args.verbose):
    print("Input dir:", args.inputDir)
    print("Output dir:", args.outputDir)

#everthing looks good. Let's loop over the html files in inputDir
xslt_to_xhtml = etree.XML('''\
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0"
   xmlns:html='http://www.w3.org/1999/xhtml'>

   <xsl:template match="*">
    <xsl:element name="html:{local-name(.)}">
      <xsl:apply-templates select="@*|*|text()"/>
       </xsl:element>
       </xsl:template>

       <xsl:template match="@*">
         <xsl:attribute name="{name(.)}"><xsl:value-of
         select="."/></xsl:attribute>
         </xsl:template>

         </xsl:stylesheet>''')
transform_to_xhtml = etree.XSLT(xslt_to_xhtml)

EXTENSIONS = ('.hocr','.html', '.htm')
XHTML_NAMESPACE = "http://www.w3.org/1999/xhtml"
for root, dirs, files in os.walk(args.inputDir):
    for file_name in files:
        if file_name.endswith(EXTENSIONS):
            print(file_name)
            with open(os.path.join(args.inputDir,file_name)) as file: # Use file to refer to the file
                try:
                    tree = etree.parse(file)
                    find_xhtml_body = etree.ETXPath("//{%s}body" % XHTML_NAMESPACE)
                    results = find_xhtml_body(tree)
                    xhtml = transform_to_xhtml(tree)
                    if (args.fixBigWordSpans):
                        try:
                            fix_word_span_area(xhtml)
                        except BboxError:
                            if (args.verbose):
                                print("we get a bbox error while trying to fix word span areas on " + file_name + " so we'll try to rewrite the page bbox using imageDir")
                            try:
                                #todo check that args.imageDir is set
                                if not(os.path.isdir(args.imageDir)):
                                    print("To correct these files, the imageDir argument has to be set to a valid directory.")
                                    exit(1)
                                #remove '.html' and add '.png'
                                image_file_name = file_name[:-5] + '.png'
                                image_path = os.path.join(args.imageDir, image_file_name)
                                image = PIL.Image.open(image_path)
                                image_x, image_y = image.size
                                rewrite_ocr_page_title(xhtml, file_name, image_x, image_y)
                                fix_word_span_area(xhtml)
                            except Exception:
                                print("something went wrong trying to fix the page bbox on this file. Aborting.")
                                raise
                    clean_ocr_page_title(xhtml, file_name)
                    if (args.shareSpaceSpans):
                        share_space_spans(xhtml)
                    if (args.confidenceSummary):
                        confidence_summary(xhtml)
                    #push_edge_spans_to_borders_of_line(xhtml)
                    xhtml.write(os.path.join(args.outputDir,file_name),pretty_print=True, xml_declaration=True,   encoding="utf-8")
                except Exception as e:
                    print("This exception was thrown on file {}".format(file_name))
                    print(e)
