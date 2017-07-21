# -*- coding: utf-8 -*-

""" Main script to collect data """

# libraries
from toolbox.utils import get_config_tag, get_config_trace
import collect_metadata
import basex_http_management
import download_data
import post_download
print("\n")

get_config_trace()

# paths
metadata_directory = get_config_tag("output", "metadata")
error_metadata_directory = get_config_tag("error", "metadata")
organization_directory = get_config_tag("output_organization", "metadata")
error_organization_directory = get_config_tag("error_organization", "metadata")
reuse_directory = get_config_tag("output_reuse", "metadata")
error_reuse_directory = get_config_tag("error_reuse", "metadata")
output_basex = get_config_tag("output", "basex")
query_directory = get_config_tag("query", "basex")
output_download = get_config_tag("output", "download")
error_download = get_config_tag("error", "download")

# parameters
reset_metadata = get_config_tag("reset", "metadata")
reset_basex = get_config_tag("reset", "basex")
n_jobs_download = get_config_tag("n_jobs", "download")
reset_download = get_config_tag("reset", "download")
format_download = get_config_tag("format", "download")
multiprocessing_download = get_config_tag("multi", "download")

# collect metadata about datasets, organizations and reuses
collect_metadata.main(output_directory=metadata_directory,
                      organization_directory=organization_directory,
                      reuse_directory=reuse_directory,
                      error_directory=error_metadata_directory,
                      error_organization_directory=error_organization_directory,
                      error_reuse_directory=error_reuse_directory,
                      reset=reset_metadata)

# execute queries on the metadata with BaseX
basex_http_management.main(input_directory=metadata_directory,
                           output_directory=output_basex,
                           query_directory=query_directory,
                           reset=reset_basex)

# download data
download_data.main(input_directory=output_basex,
                   output_directory=output_download,
                   error_directory=error_download,
                   n_jobs=n_jobs_download,
                   reset=reset_download,
                   format=format_download,
                   multi=multiprocessing_download)

post_download.main(input_directory=output_download,
                   basex_directory=output_basex,
                   error_directory=error_download)
