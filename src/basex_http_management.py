# -*- coding: utf-8 -*-

# libraries
import os
import re
import requests
from requests import adapters
import subprocess
import time
from lxml import etree
from urllib.parse import urljoin
from toolbox.utils import TryMultipleTimes, get_config_tag
print("\n")

# http://docs.basex.org/wiki/REST


def run_query_file(query_file):
    return _run_file(query_file, post_query_template)


def run_command_file(command_file):
    return _run_file(command_file, post_command_template)


def execute_query(query):
    return _post_request(query, template=post_query_template)


def execute_command(command):
    return _post_request(command, template=post_command_template)


def _run_file(file_to_run, template):
    return _send_file_to_run(file_to_run, template)


def _send_file_to_run(file_to_run, template):
    with open(file_to_run) as file_h:
        instructions = file_h.read()
        return _post_request(instructions, template)


@TryMultipleTimes()
def _post_request(body, template):
    assert not re.search(r'\]\s*\]\s*>', body)
    body = template.format(body=body)
    return requests_session.post(server_url, data=body.encode('utf-8')).text

# path
metadata_directory = get_config_tag("input", "basex")
output_path = get_config_tag("output", "basex")
queries_directory = os.getcwd()

# check output directory exists
if not os.path.isdir(os.path.dirname(output_path)):
    os.mkdir(os.path.dirname(output_path))

# http backend configuration
requests_session = requests.Session()
adapter = adapters.HTTPAdapter(max_retries=5)
requests_session.mount('http://', adapter)
requests_session.auth = ("admin", "admin")
database_name = "metadata"
server_url = "http://localhost:8984/rest/"
database_url = urljoin(server_url, database_name)
basex_directory = os.path.join(os.path.expanduser("~/"), "basex", "bin")
url_destination_condition = "where $url_destination = 'file'"
# list_url_destination = ["file", "remote", "api"]

post_query_template = (
    """
    <rest:query xmlns:rest="http://basex.org/rest">
    <rest:text><![CDATA[ {body} ]]></rest:text>
    </rest:query>
    """)

post_command_template = (
    """
    <rest:command xmlns:rest="http://basex.org/rest">
    <rest:text><![CDATA[ {body} ]]></rest:text>
    </rest:command>
    """)

# launch the server
server = subprocess.Popen(os.path.join(basex_directory, "basexhttp"),
                          stdout=subprocess.PIPE)
time.sleep(5)

# create database
print(execute_command("CREATE DB metadata " + metadata_directory))
print(execute_command("list"))

# TODO allow query with different formats (xls, csv, zip, etc.)
# TODO allow query with different kind of file (remote, API, file)
# query
print(os.path.join(queries_directory, "total_query.xq"))
x = run_query_file(os.path.join(queries_directory, "total_query.xq"))
tree = etree.fromstring(x)
url_list = [url.text for url in tree.xpath("/results/table/url")]
print("number of files :", len(url_list), "\n")

# save the xml
obj_xml = etree.tostring(tree, pretty_print=True, xml_declaration=True,
                         encoding="UTF-8")
with open(output_path, "wb") as xml_writer:
    xml_writer.write(obj_xml)

# drop database
print(execute_command("DROP database metadata"))

# stop the server
subprocess.call(os.path.join(basex_directory, "basexhttpstop"))


def main():
    return

if __name__ == "__main__":

    # paths
    input_dataset = get_config_tag("output", "metadata")
    input_organization = get_config_tag("output_organization", "metadata")
    input_reuse = get_config_tag("output_reuse", "metadata")
    output_directory = get_config_tag("output", "basex")





    # run
    main()
