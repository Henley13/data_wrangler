#!/bin/python3
# coding: utf-8

""" We parse and collect information from the xml files."""

# libraries
import subprocess
import time
from lxml import etree
from BaseXClient import BaseXClient
print("\n")

# variables
directory = "../data/metadata/"
filename = "../url_csv.xml"
format_condition = "where $format = ('CSV', 'csv')"
# list_format = ["CSV", "csv", "JSON", "json", "ZIP", "SHP", "shp", "XLS", "xls", "HTML"]
url_destination_condition = "where $url_destination = 'file'"
# list_url_destination = ["file", "remote", "api"]

# launch the server
no_server = True
server = subprocess.Popen("basexserver", stdout=subprocess.PIPE)
while no_server:
    time.sleep(1)
    for line in server.stdout:
        if "Server was started" in line.decode("utf-8"):
            no_server = False
            break
print("server launched", "\n")

# open session
session = BaseXClient.Session("localhost", 1984, 'admin', 'admin')
print("session opened...", "\n")

# create database
session.execute("CREATE DB metadata /home/arthur/arthur_imbert/python/data/metadata/")
print(session.info())

# build and execute the xquery
l = list()
#####################################################################
l.append("xquery")
l.append("<results> {")
l.append("for $table in collection('/metadata')//tables/table")
l.append("let $id := $table/id")
l.append("let $url := $table/url")
l.append("let $url_destination := $table/url_destination")
l.append("let $format := $table/format")
l.append(format_condition)
l.append(url_destination_condition)
l.append("return <table>{$url, $id}</table>")
l.append("} </results>")
#####################################################################
xq = " ".join(l)
tree = etree.fromstring(session.execute(xq))
url_list = [url.text for url in tree.xpath("/results/table/url")]
print("nombre de tables :", len(url_list), "\n")

# save the xml
obj_xml = etree.tostring(tree, pretty_print=True, xml_declaration=True, encoding="UTF-8")
with open(filename, "wb") as xml_writer:
    xml_writer.write(obj_xml)

# drop database
session.execute("drop db metadata")
print(session.info())

# close the session
session.close()
print("...session closed!")

# stop the server
subprocess.call(["basexserver", "stop"])
