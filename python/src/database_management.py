#!/bin/python3
# coding: utf-8

""" Parse and collect information from the xml files."""

# libraries
from BaseXClient import BaseXClient

# create session
session = BaseXClient.Session("127.0.0.1", 1984, 'admin', 'admin')

if session:
    session.close()


stopgo = False
if stopgo:
    try:
        # create new database
        session.create("database", "<x>Hello World!</x>")
        print(session.info())

        # run query on database
        print("\n" + session.execute("xquery doc('database')"))

        # drop database
        session.execute("drop db database")
        print(session.info())

    finally:
        # close session
        if session:
            session.close()

