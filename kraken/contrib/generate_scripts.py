#!/usr/bin/env python3
"""
Script fetching the latest unicode Scripts.txt and dumping it as json.
"""
from urllib import request
import json
import regex

uri = 'http://www.unicode.org/Public/UNIDATA/Scripts.txt'

re = regex.compile('^(?P<start>[0-9A-F]{4,6})(..(?P<end>[0-9A-F]{4,6}))?\s+; (?P<name>[A-Za-z]+)')

with open('scripts.json', 'w') as fp, request.urlopen(uri) as req:
    d = []
    for line in req:
        line = line.decode('utf-8')
        if line.startswith('#') or line.strip() == '':
            continue
        m = re.match(line)
        if m:
            print(line)
            start = int(m.group('start'), base=16)
            end = start
            if m.group('end'):
                end = int(m.group('end'), base=16)
            name = m.group('name')
            if len(d) > 0 and d[-1][2] == name and (start - 1 == d[-1][1] or start -1 == d[-1][0]):
                print('merging {} and ({}, {}, {})'.format(d[-1], start, end, name))
                d[-1] = (d[-1][0], end, name)
            else:
                d.append((start, end if end != start else None, name))
    json.dump(d, fp)
