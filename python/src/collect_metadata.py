#!/bin/python3
# coding: utf-8

"""We collect metadata on data.gouv.fr about numerous datasets and store it in a xml format."""

# libraries
import os
import requests
import math
import sys
import traceback
from lxml import etree, objectify
print("\n")

# variables
directory = "../data/metadata/"
# url_api = "https://www.data.gouv.fr/api/1/datasets/?page_size=10&page=1"
url_api = "https://www.data.gouv.fr/api/1/datasets/?page_size=50&page=1&format="
formats_file = ["CSV", "csv", "JSON", "json", "SHP", "shp", "ZIP", "XLS", "xls", "HTML"]
d_formats_file = {}
datasets_collected_total = 0
errors = []

# Functions


def create_badges(badges):
    """
    Create a badges XML element.
    :param badges: list of dictionaries
    :return: ObjectifiedElement
    """
    badges_e = objectify.Element("badges")
    for badge in badges:
        badges_e.addattr("badge", badge["kind"])
    return badges_e


def create_organization(organization):
    """
    Create an organization XML element.
    :param organization: dict
    :return: ObjectifiedElement
    """
    organization_e = objectify.Element("organization")
    if isinstance(organization, dict):
        organization_e.title = organization["name"]
        organization_e.id = organization["id"]
        organization_e.url = organization["page"]
        organization_e.url_api = organization["uri"]
    else:
        organization_e.title = None
        organization_e.id = None
        organization_e.url = None
        organization_e.url_api = None
    return organization_e


def create_tags(tags):
    """
    Create a tags XML element.
    :param tags: list
    :return: ObjectifiedElement
    """
    tags_e = objectify.Element("tags")
    for tag in tags:
        tags_e.addattr("tag", tag)
    return tags_e


def create_tables(tables):
    """
    Create a tables XML element.
    :param tables: list of dictionaries
    :return: ObjectifiedElement
    """
    tables_e = objectify.Element("tables")
    for table in tables:
        table_e = objectify.SubElement(tables_e, "table")
        table_e.title = table["title"]
        table_e.id = table["id"]
        table_e.url = table["url"]
        table_e.description = table["description"]
        table_e.creation = table["created_at"]
        table_e.publication = table["published"]
        table_e.last_modification = table["last_modified"]
        table_e.size = table["filesize"]
        table_e.format = table["format"]
        table_e.url_destination = table["filetype"]
        table_e.availability = table["is_available"]
    return tables_e


def create_xml(data):
    """
    Create an XML file
    :param data: dict
    :return:
    """
    xml = '''<?xml version="1.0" encoding="UTF-8"?><metadata></metadata>'''
    root = objectify.fromstring(xml.encode('UTF-8'))
    # general information
    root.id = data["id"]
    root.title = data["title"]
    root.url = data["page"]
    root.url_api = data["uri"]
    root.license = data["license"]
    root.description = data["description"]
    root.creation = data["created_at"]
    root.last_modification = data["last_modified"]
    root.last_resources_update = data["last_update"]
    # temporal metadata
    root.frequency = data["frequency"]
    if isinstance(data["temporal_coverage"], dict):
        root.start_coverage = data["temporal_coverage"]["start"]
        root.end_coverage = data["temporal_coverage"]["end"]
    else:
        root.start_coverage = None
        root.end_coverage = None
    # spatial metadata
    if isinstance(data["spatial"], dict):
        root.granularity = data["spatial"]["granularity"]
        root.zones = data["spatial"]["zones"]
        root.geojson = data["spatial"]["geom"]
    else:
        root.granularity = None
        root.zones = None
        root.geojson = False
    # producer metadata
    root.append(create_organization(data["organization"]))
    # miscellaneous metadata
    root.append(create_badges(data["badges"]))
    root.append(create_tags(data["tags"]))
    # table metadata
    root.append(create_tables(data["resources"]))
    # remove lxml annotation
    objectify.deannotate(root)
    etree.cleanup_namespaces(root)
    # create the xml string
    obj_xml = etree.tostring(root, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    # save the xml
    filename = directory + data["id"] + ".xml"
    with open(filename, "wb") as xml_writer:
        xml_writer.write(obj_xml)
    return

# check if the output directory exists
if not os.path.isdir(directory):
    os.mkdir(directory)
    print(directory, "created")
# get general data from data.gouv API
r = requests.get("https://www.data.gouv.fr/api/1/datasets/?page_size=1&page=1")
d = r.json()
n_datasets = d["total"]
r = requests.get(url_api)
d = r.json()
n_datasets_loaded = d["total"]
page_size = d["page_size"]
print("total number of datasets :", n_datasets)
print("datasets per page :", page_size)
print("\n")

# get data from data.gouv API and iterate over the different pages
for format_file in formats_file:
    url_api_format = url_api + format_file
    r = requests.get(url_api_format)
    d = r.json()
    n_datasets_loaded = d["total"]
    datasets_collected = 0
    print("total number of loaded datasets :", n_datasets_loaded)
    print("page to load :", math.ceil(n_datasets_loaded / page_size))
    print("starting url :", r.url)
    print("\n")
    while datasets_collected < n_datasets_loaded:
        # load the json
        r = requests.get(url_api_format)
        if r.status_code == 500:
            break
        d = r.json()
        page = d["page"]
        page_size = d["page_size"]
        url_api_format = d["next_page"]
        print("----- processing page", page, "-----")
        # build an XML per dataset included all metadata needed
        for data in d["data"]:
            try:
                create_xml(data)
                datasets_collected += 1
            except ValueError:
                exc_info = sys.exc_info()
                print("ValueError :", data["page"])
                traceback.print_exception(*exc_info)
                n_datasets_loaded -= 1
                errors.append(data["page"])
                pass
    datasets_collected_total += datasets_collected
    d_formats_file[format_file] = datasets_collected
    print("\n")

print("-------------------------------------------------------------")
print("errors :", len(errors))
with open('../data/APIerrors.txt', mode='wt', encoding='utf-8') as f:
    f.write('\n'.join(errors))
print(datasets_collected_total, "(meta)collected datasets (", round(datasets_collected_total / n_datasets * 100, 2), "% )")
for i in d_formats_file:
    print(i, ":", d_formats_file[i], "files, (", round(d_formats_file[i] / n_datasets * 100, 2), "% )")
