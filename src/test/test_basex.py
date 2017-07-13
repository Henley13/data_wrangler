# -*- coding: utf-8 -*-

# libraries
import os
from test.utils import assert_equal
from test.utils import assert_true
from test.utils import assert_greater
from basex_http_management import backend_configuration, BaseXServer, get_url
print("\n")


# TODO test how to update database
# TODO test when you launch and stop the server
# TODO be sure the server is down when we start the test
# TODO improve backend testing
# TODO test content of the output
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
    url = "http://{hostname}:8984/rest/".format(hostname=hostname)
    assert_equal(server_url, url)

    # test session configuration
    assert_equal(requests_session.auth, ('admin', 'admin'))

    return


def test_get_url():
    # initialize server
    (server_url, basex_directory, post_query_template, post_command_template,
     session, hostname) = backend_configuration()

    # initialized BaseX server
    basex = BaseXServer(session, server_url, hostname, post_query_template,
                        post_command_template, basex_directory)

    # collect urls
    current_script_directory = os.path.dirname(os.path.abspath(__file__))
    database_name = "test_database"
    input_test_dataset = os.path.join(current_script_directory, "data/metadata")
    output_test_dataset = os.path.join(current_script_directory, "data")
    query_test_directory = os.path.join(current_script_directory, "query")
    get_url(basex, input_test_dataset, output_test_dataset,
            query_test_directory, database_name)

    # test output file exists and is not empty
    output_path = os.path.join(output_test_dataset, "url_test.xml")
    assert_true(os.path.isfile(output_path))
    assert_greater(os.path.getsize(output_path), 0)
    os.remove(output_path)

    return
