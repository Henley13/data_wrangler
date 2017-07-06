# -*- coding: utf-8 -*-

""" We collect metadata from data.gouv.fr about numerous datasets and store it
    in a xml format. """

# libraries
import os
import requests
import shutil
from tqdm import tqdm
from lxml import etree, objectify
from toolbox.utils import log_error, get_config_tag
print("\n")


def _check_error_directory(error_directory, error_organization_directory,
                           error_reuse_directory, reset):
    """
    Function to check and reset the error directories
    :param error_directory: string
    :param error_organization_directory: string
    :param error_reuse_directory: string
    :return:
    """
    if not os.path.isdir(os.path.dirname(error_directory)):
        os.mkdir(os.path.dirname(error_directory))
    # error_directory
    if not os.path.isdir(error_directory):
        os.mkdir(error_directory)
    elif reset:
        shutil.rmtree(error_directory)
        os.mkdir(error_directory)
    else:
        pass
    # error_organization_directory
    if not os.path.isdir(error_organization_directory):
        os.mkdir(error_organization_directory)
    elif reset:
        shutil.rmtree(error_organization_directory)
        os.mkdir(error_organization_directory)
    else:
        pass
    # error_reuse_directory
    if not os.path.isdir(error_reuse_directory):
        os.mkdir(error_reuse_directory)
    elif reset:
        shutil.rmtree(error_reuse_directory)
        os.mkdir(error_reuse_directory)
    else:
        pass
    return


def _check_output_directory(output_directory, organization_directory,
                            reuse_directory, reset):
    """
    Function to check and reset the output directories
    :param output_directory: string
    :param organization_directory: string
    :param reuse_directory: string
    :param reset: boolean
    :return:
    """
    # output_directory
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)
    elif reset:
        shutil.rmtree(output_directory)
        os.mkdir(output_directory)
    else:
        pass
    # organization_directory
    if not os.path.isdir(organization_directory):
        os.mkdir(organization_directory)
    elif reset:
        shutil.rmtree(organization_directory)
        os.mkdir(organization_directory)
    else:
        pass
    # reuse_directory
    if not os.path.isdir(reuse_directory):
        os.mkdir(reuse_directory)
    elif reset:
        shutil.rmtree(reuse_directory)
        os.mkdir(reuse_directory)
    else:
        pass
    return


def _create_badges(badges):
    """
    Create a badges XML element.
    :param badges: list of dictionaries
    :return: ObjectifiedElement
    """
    badges_e = objectify.Element("badges")
    for badge in badges:
        badges_e.addattr("badge", badge["kind"])
    return badges_e


def _create_organization(organization):
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


def _create_tags(tags):
    """
    Create a tags XML element.
    :param tags: list
    :return: ObjectifiedElement
    """
    tags_e = objectify.Element("tags")
    for tag in tags:
        tags_e.addattr("tag", tag)
    return tags_e


def _create_tables(tables):
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
        table_e.creation = table["created_at"]
        table_e.publication = table["published"]
        table_e.last_modification = table["last_modified"]
        table_e.size = table["filesize"]
        table_e.format = table["format"]
        table_e.url_destination = table["filetype"]
        table_e.availability = table["is_available"]
        try:
            table_e.description = table["description"]
        except ValueError:
            table_e.description = None
    return tables_e


def _create_datasets(datasets):
    """
    Create a datasets XML element.
    :param datasets: list of dictionaries
    :return: ObjectifiedElement
    """
    datasets_e = objectify.Element("datasets")
    for dataset in datasets:
        dataset_e = objectify.SubElement(datasets_e, "dataset")
        dataset_e.title = dataset["title"]
        dataset_e.id = dataset["id"]
        dataset_e.url_api = dataset["uri"]
        dataset_e.url = dataset["page"]
    return datasets_e


def _create_xml_metadata(data, output_directory):
    """
    Create an XML file for the metadata
    :param data: dictionary
    :param output_directory: string
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
    root.creation = data["created_at"]
    root.last_modification = data["last_modified"]
    root.last_resources_update = data["last_update"]
    try:
        root.description = data["description"]
    except ValueError:
        root.description = None

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
    root.append(_create_organization(data["organization"]))

    # miscellaneous metadata
    root.append(_create_badges(data["badges"]))
    root.append(_create_tags(data["tags"]))

    # table metadata
    root.append(_create_tables(data["resources"]))

    # metrics metadata
    if isinstance(data["metrics"], dict):
        root.discussions = data["metrics"]["discussions"]
        root.followers = data["metrics"]["followers"]
        root.views = data["metrics"]["views"]
        root.nb_uniq_visitors = data["metrics"]["nb_uniq_visitors"]
        root.nb_visits = data["metrics"]["nb_visits"]
        root.nb_hits = data["metrics"]["nb_hits"]
        root.reuses = data["metrics"]["reuses"]
    else:
        root.discussions = None
        root.followers = None
        root.views = None
        root.nb_uniq_visitors = None
        root.nb_visits = None
        root.nb_hits = None
        root.reuses = None

    # remove lxml annotation
    objectify.deannotate(root)
    etree.cleanup_namespaces(root)

    # create the xml string
    obj_xml = etree.tostring(root, pretty_print=True, xml_declaration=True,
                             encoding="utf-8")

    # save the xml
    filepath = os.path.join(output_directory, data["id"] + ".xml")
    with open(filepath, "wb") as xml_writer:
        xml_writer.write(obj_xml)

    return


def _create_xml_organization(data, organization_directory):
    """
    Create an XML file for the organization metadata
    :param data: dictionary
    :param organization_directory: string
    :return:
    """
    xml = '''<?xml version="1.0" encoding="UTF-8"?><metadata></metadata>'''
    root = objectify.fromstring(xml.encode('UTF-8'))

    # general information
    root.id = data["id"]
    root.title = data["name"]
    root.url = data["page"]
    root.url_api = data["uri"]
    try:
        root.description = data["description"]
    except ValueError:
        root.description = None

    # metrics metadata
    if isinstance(data["metrics"], dict):
        root.dataset_views = data["metrics"]["dataset_views"]
        root.datasets = data["metrics"]["datasets"]
        root.followers = data["metrics"]["followers"]
        root.members = data["metrics"]["members"]
        root.resource_downloads = data["metrics"]["resource_downloads"]
        root.reuse_views = data["metrics"]["reuse_views"]
        root.reuses = data["metrics"]["reuses"]
        root.views = data["metrics"]["views"]
        root.nb_uniq_visitors = data["metrics"]["nb_uniq_visitors"]
        root.nb_visits = data["metrics"]["nb_visits"]
        root.permitted_reuses = data["metrics"]["permitted_reuses"]
        root.nb_hits = data["metrics"]["nb_hits"]
    else:
        root.dataset_views = None
        root.datasets = None
        root.followers = None
        root.members = None
        root.resource_downloads = None
        root.reuse_views = None
        root.reuses = None
        root.views = None
        root.nb_uniq_visitors = None
        root.nb_visits = None
        root.permitted_reuses = None
        root.nb_hits = None

    # remove lxml annotation
    objectify.deannotate(root)
    etree.cleanup_namespaces(root)

    # create the xml string
    obj_xml = etree.tostring(root, pretty_print=True, xml_declaration=True,
                             encoding="utf-8")
    # save the xml
    filepath = os.path.join(organization_directory, data["id"] + ".xml")
    with open(filepath, "wb") as xml_writer:
        xml_writer.write(obj_xml)

    return


def _create_xml_reuse(data, reuse_directory):
    """
    Create an XML file for the reuse metadata
    :param data: dictionary
    :param reuse_directory: string
    :return:
    """
    xml = '''<?xml version="1.0" encoding="UTF-8"?><metadata></metadata>'''
    root = objectify.fromstring(xml.encode('UTF-8'))

    # general information
    root.id = data["id"]
    root.title = data["title"]
    root.url = data["page"]
    root.url_api = data["uri"]
    root.creation = data["created_at"]
    root.last_modification = data["last_modified"]
    root.type = data["type"]
    try:
        root.description = data["description"]
    except ValueError:
        root.description = None
    # tags metadata
    root.append(_create_tags(data["tags"]))

    # metrics metadata
    if isinstance(data["metrics"], dict):
        root.datasets = data["metrics"]["datasets"]
        root.followers = data["metrics"]["followers"]
        root.views = data["metrics"]["views"]
        root.nb_uniq_visitors = data["metrics"]["nb_uniq_visitors"]
        root.nb_visits = data["metrics"]["nb_visits"]
        root.nb_hits = data["metrics"]["nb_hits"]
    else:
        root.datasets = None
        root.followers = None
        root.views = None
        root.nb_uniq_visitors = None
        root.nb_visits = None
        root.nb_hits = None

    # datasets metadata
    root.append(_create_datasets(data["datasets"]))

    # remove lxml annotation
    objectify.deannotate(root)
    etree.cleanup_namespaces(root)

    # create the xml string
    obj_xml = etree.tostring(root, pretty_print=True, xml_declaration=True,
                             encoding="utf-8")

    # save the xml
    filepath = os.path.join(reuse_directory, data["id"] + ".xml")
    with open(filepath, "wb") as xml_writer:
        xml_writer.write(obj_xml)

    return


def collect_organization_metadata(organization_directory,
                                  error_organization_directory):
    """
    Function to collect metadata about organizations
    :param organization_directory: string
    :param error_organization_directory: string
    :return: list of strings
    """
    print("collect organizations' metadata", "\n")

    # initialization
    id_organizations = []
    organizations_collected_total = 0
    n_errors = 0
    url_organization = ("https://www.data.gouv.fr/api/1/organizations/"
                        "?sort=-datasets&page_size=50&page=1")

    # get the total number of organizations
    r = requests.get("https://www.data.gouv.fr/api/1/organizations/?"
                     "sort=-datasets&page_size=1&page=1")
    d = r.json()
    n_organizations = d["total"]
    print("total number of organizations :", n_organizations)
    print("\n")

    # collect ids
    while organizations_collected_total < n_organizations:
        # load the json
        r = requests.get(url_organization)
        if r.status_code == 500:
            break
        d = r.json()
        url_organization = d["next_page"]
        # get the id and collect the metadata
        for data in d["data"]:
            try:
                id = data["id"]
                id_organizations.append(id)
                _create_xml_organization(data, organization_directory)
                organizations_collected_total += 1
            except (ValueError, UnicodeDecodeError):
                path = os.path.join(error_organization_directory, data["id"])
                log_error(path, [data["page"]])
                n_errors += 1
                organizations_collected_total += 1
                pass

    organizations_collected_total -= n_errors
    print("number of organizations collected :", organizations_collected_total,
          "(", round(organizations_collected_total / n_organizations * 100, 2),
          "% )")
    # check if all the organizations have been correctly collected
    if len(id_organizations) != n_organizations:
        raise Exception("organizations are missing : %i have been collected, "
                        "expected %i"
                        % (len(id_organizations), n_organizations))

    print("number of errors :", n_errors, "\n")

    return id_organizations


def collect_metadata(output_directory, error_directory, id_organizations):
    """
    Function to collect metadata about datasets
    :param output_directory: string
    :param error_directory: string
    :param id_organizations: list of strings
    :return:
    """
    print("collect datasets' metadata", "\n")

    # initialization
    n_errors = 0
    datasets_collected_total = 0
    url_api_template = ("https://www.data.gouv.fr/api/1/datasets/?page_size=50&"
                        "page=1&organization=")

    # get the total number of datasets
    r = requests.get("https://www.data.gouv.fr/api/1/datasets/"
                     "?page_size=1&page=1")
    d = r.json()
    n_datasets = d["total"]
    print("total number of datasets :", n_datasets)
    print("\n")

    # collect metadata and iterate over the different pages
    for id_organization in tqdm(id_organizations):
        url_api_organization = url_api_template + str(id_organization)
        r = requests.get(url_api_organization)
        d = r.json()
        n_datasets_loaded = d["total"]
        datasets_collected = 0
        while datasets_collected < n_datasets_loaded:
            # load the json
            r = requests.get(url_api_organization)
            if r.status_code == 500:
                break
            d = r.json()
            url_api_organization = d["next_page"]
            # build an XML per dataset included all metadata needed
            for data in d["data"]:
                try:
                    _create_xml_metadata(data, output_directory)
                    datasets_collected += 1
                except (ValueError, UnicodeDecodeError):
                    path = os.path.join(error_directory, data["id"])
                    log_error(path, [data["page"]])
                    n_errors += 1
                    datasets_collected += 1
                    pass
        datasets_collected_total += datasets_collected

    datasets_collected_total -= n_errors
    print("number of datasets collected :", datasets_collected_total,
          "(", round(datasets_collected_total / n_datasets * 100, 2), "% )")
    print("number of errors :", n_errors, "\n")

    return


def collect_reuse_metadata(reuse_directory, error_reuse_directory):
    """
    Function to collect metadata about reuses
    :param reuse_directory: string
    :param error_reuse_directory: string
    :return:
    """
    print("collect reuses' metadata", "\n")

    # initialization
    reuses_collected_total = 0
    n_errors = 0
    url_reuse = "http://www.data.gouv.fr/api/1/reuses/?page_size=20"

    # get the total number of reuses
    r = requests.get("http://www.data.gouv.fr/api/1/reuses/?page_size=1")
    d = r.json()
    n_reuses = d["total"]
    print("total number of reuses :", n_reuses)
    print("\n")

    while reuses_collected_total < n_reuses:
        # load the json
        r = requests.get(url_reuse)
        if r.status_code == 500:
            break
        d = r.json()
        url_reuse = d["next_page"]
        # get the id and collect the metadata
        for data in d["data"]:
            try:
                _create_xml_reuse(data, reuse_directory)
                reuses_collected_total += 1
            except (ValueError, UnicodeDecodeError):
                path = os.path.join(error_reuse_directory, data["id"])
                log_error(path, [data["page"]])
                n_errors += 1
                reuses_collected_total += 1
                pass

    reuses_collected_total -= n_errors
    print("number of reuses collected :", reuses_collected_total,
          "(", round(reuses_collected_total / n_reuses * 100, 2), "% )")

    print("number of errors :", n_errors, "\n")

    return


def main(output_directory, organization_directory, reuse_directory,
         error_directory, error_organization_directory, error_reuse_directory,
         reset):
    """
    Function to run all the script
    :param output_directory: string
    :param organization_directory: string
    :param reuse_directory: string
    :param error_directory: string
    :param error_organization_directory: string
    :param error_reuse_directory:string
    :param reset: boolean
    :return:
    """
    print("collect metadata", "\n")

    # check the error directories
    _check_error_directory(error_directory, error_organization_directory,
                           error_reuse_directory, reset)

    # check the output directories
    _check_output_directory(output_directory, organization_directory,
                            reuse_directory, reset)

    # collect metadata about organizations
    id_org = collect_organization_metadata(organization_directory,
                                           error_organization_directory)

    # collect metadata about datasets
    collect_metadata(output_directory, error_directory, id_org)

    # collect metadata about reuses
    collect_reuse_metadata(reuse_directory, error_reuse_directory)

    return

if __name__ == "__main__":

    # paths
    output_directory = get_config_tag("output", "metadata")
    error_directory = get_config_tag("error", "metadata")
    organization_directory = get_config_tag("output_organization", "metadata")
    error_organization_directory = get_config_tag("error_organization",
                                                  "metadata")
    reuse_directory = get_config_tag("output_reuse", "metadata")
    error_reuse_directory = get_config_tag("error_reuse", "metadata")

    # parameters
    reset = get_config_tag("reset", "metadata")

    # run
    main(output_directory=output_directory,
         organization_directory=organization_directory,
         reuse_directory=reuse_directory,
         error_directory=error_directory,
         error_organization_directory=error_organization_directory,
         error_reuse_directory=error_reuse_directory,
         reset=reset)
