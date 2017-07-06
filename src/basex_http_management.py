# -*- coding: utf-8 -*-

# libraries
import os
import re
import shutil
import subprocess
import time
import glob
from requests import Session, adapters
from lxml import etree
from urllib.parse import urljoin
from toolbox.utils import TryMultipleTimes, get_config_tag
print("\n")

# http://docs.basex.org/wiki/REST


def _check_output_directory(output_directory, reset):
    """
    Function to check the output directory
    :param output_directory: string
    :param reset: boolean
    :return:
    """
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)
    else:
        if reset:
            shutil.rmtree(output_directory)
            os.mkdir(output_directory)
        else:
            pass
    return


def backend_configuration():
    requests_session = Session()
    adapter = adapters.HTTPAdapter(max_retries=5)
    requests_session.mount('http://', adapter)
    requests_session.auth = ("admin", "admin")
    server_url = "http://localhost:8984/rest/"
    basex_directory = os.path.join(os.path.expanduser("~/"), "basex", "bin")
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

    return (server_url, basex_directory, post_query_template,
            post_command_template, requests_session)


class BaseXServer:

    def __init__(self, session, url, query_template, command_template,
                 basex_directory):
        self.session = session
        self.server_url = url
        self.query_template = query_template
        self.command_template = command_template
        self.basex_directory = basex_directory
        self.database_url = None
        self.database_name = None

    @TryMultipleTimes()
    def post_request(self, body, template):
        assert not re.search(r'\]\s*\]\s*>', body)
        body = template.format(body=body)
        return self.session.post(self.server_url,
                                 data=body.encode('utf-8')).text

    def send_file_to_run(self, file_to_run, template):
        with open(file_to_run) as file_h:
            instructions = file_h.read()
            return self.post_request(body=instructions, template=template)

    def run_query_file(self, query_file):
        return self.send_file_to_run(file_to_run=query_file,
                                     template=self.query_template)

    def run_command_file(self, command_file):
        return self.send_file_to_run(file_to_run=command_file,
                                     template=self.command_template)

    def execute_query(self, query):
        return self.post_request(body=query, template=self.query_template)

    def execute_command(self, command):
        return self.post_request(body=command, template=self.command_template)

    def launch_server(self):
        """
        Function to start a BaseX http server.

        """
        subprocess.Popen(os.path.join(self.basex_directory, "basexhttp"),
                         stdout=subprocess.PIPE)
        time.sleep(5)
        return

    def create_database(self, name_database, input_directory):
        """
        Function to create a database in BaseX from a multiple XML files.

        Parameters
        ----------
        name_database : str
            Name of the database

        input_directory : str
            Path of the XML directory

        Returns
        -------

        """
        command = "CREATE DB {database} {input}".format(database=name_database,
                                                        input=input_directory)
        self.execute_command(command)
        self.execute_command("list")
        self.database_name = name_database
        self.database_url = urljoin(self.server_url, name_database)
        return

    def drop_database(self, name_database):
        """
        Function to drop a database.

        Parameters
        ----------
        name_database : str
            Name of the database in BaseX

        Returns
        -------

        """
        command = "DROP database {database}".format(database=name_database)
        self.execute_command(command)
        self.database_name = None
        self.database_url = None
        return

    def stop_server(self):
        """
        Function to stop the BaseX server.

        """
        subprocess.call(os.path.join(self.basex_directory, "basexhttpstop"))
        time.sleep(5)
        return


def get_url(server, input_dataset, output_directory, query_directory,
            database_name):

    print("collect urls", "\n")

    # launch the server
    server.launch_server()

    # create database
    server.create_database(database_name, input_dataset)

    # queries
    for query_file in glob.glob(query_directory):
        path_query = os.path.join(query_directory, query_file)
        format_files = query_file.split("_")[0]
        x = server.run_query_file(path_query)
        tree = etree.fromstring(x)
        url_list = [url.text for url in tree.xpath("/results/table/url")]
        print(format_files, ": number of files :", len(url_list))
        # save the xml
        obj_xml = etree.tostring(tree, pretty_print=True, xml_declaration=True,
                                 encoding="utf-8")
        output_path = os.path.join(output_directory,
                                   "url_" + format_files + ".xml")
        with open(output_path, "wb") as xml_writer:
            xml_writer.write(obj_xml)

    # drop database
    server.drop_database(database_name)

    # stop the server
    server.stop_server()

    print()

    return


def main(input_directory, output_directory, query_directory, reset):

    # check output directory
    _check_output_directory(output_directory, reset)

    # get backend configurations
    (server_url, basex_directory, post_query_template, post_command_template,
     session) = backend_configuration()

    # initialized BaseX server
    basex = BaseXServer(session, server_url, post_query_template,
                        post_command_template, basex_directory)

    # get urls
    get_url(basex, input_directory, output_directory, query_directory,
            "metadata_dataset")

    return


if __name__ == "__main__":

    # paths
    input_dataset = get_config_tag("output", "metadata")
    input_organization = get_config_tag("output_organization", "metadata")
    input_reuse = get_config_tag("output_reuse", "metadata")
    output_directory = get_config_tag("output", "basex")
    query_directory = get_config_tag("query", "basex")

    # parameters
    reset = get_config_tag("reset", "basex")

    # run
    main(input_directory=input_dataset,
         output_directory=output_directory,
         query_directory=query_directory,
         reset=reset)
