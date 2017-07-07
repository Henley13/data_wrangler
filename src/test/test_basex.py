# -*- coding: utf-8 -*-

# libraries
import os
from test.utils import assert_equal
from test.utils import assert_true
from test.utils import assert_false
from test.utils import assert_greater
from basex_http_management import backend_configuration, BaseXServer, get_url
print("\n")


# TODO improve backend testing
def test_backend():
    (server_url, basex_directory, post_query_template, post_command_template,
     requests_session, hostname) = backend_configuration()

    # test the templates
    test_query = post_query_template.format(body="test")
    test_command = post_command_template.format(body="test")
    query = (
        """
        <rest:query xmlns:rest="http://basex.org/rest">
        <rest:text><![CDATA[ test ]]></rest:text>
        </rest:query>
        """)
    command = (
        """
        <rest:command xmlns:rest="http://basex.org/rest">
        <rest:text><![CDATA[ test ]]></rest:text>
        </rest:command>
        """)
    assert_equal(test_query, query)
    assert_equal(test_command, command)

    # test server url
    url = "http://{hostname}/rest/".format(hostname=hostname)
    assert_equal(server_url, url)

    # test session configuration
    assert_equal(requests_session.auth, ('admin', 'admin'))

    return


# TODO test how to update database
def test_basex_server():
    # initialize server
    (server_url, basex_directory, post_query_template, post_command_template,
     session, hostname) = backend_configuration()

    server = BaseXServer(session, server_url, hostname, post_query_template,
                         post_command_template, basex_directory)

    # test check_server
    assert_false(server.check_server())
    server.launch_server()
    assert_true(server.check_server())

    # test check_database
    assert_false(server.check_database())
    database_name = "test_database"
    input_test_dataset = "./data/metadata"
    server.create_database(database_name, input_test_dataset)
    assert_true(server.check_database())

    # test stop_server
    server.stop_server()
    assert_false(server.check_server())

    # test database still exist
    server.launch_server()
    assert_true(server.check_database())

    # check drop_database
    server.drop_database(database_name)
    assert_false(server.check_database())

    server.stop_server()

    return


# TODO test content of the output
def test_get_url():
    # initialize server
    (server_url, basex_directory, post_query_template, post_command_template,
     session, hostname) = backend_configuration()

    # collect urls
    database_name = "test_database"
    input_test_dataset = "./data/metadata"
    output_test_dataset = "./data"
    query_test_directory = "./query"
    get_url(session, input_test_dataset, output_test_dataset,
            query_test_directory, database_name)

    # test output file exists and is not empty
    output_path = os.path.join(output_test_dataset, "url_test.xml")
    assert_true(os.path.isfile(output_path))
    assert_greater(os.path.getsize(output_path), 0)

    return