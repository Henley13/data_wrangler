# -*- coding: utf-8 -*-

""" Main script to analyze data """

# libraries
from toolbox.utils import get_config_tag, get_config_trace
import collect_metadata
print("\n")

get_config_trace()

# paths
output_directory = get_config_tag("output", "metadata")
error_directory = get_config_tag("error", "metadata")
organization_directory = get_config_tag("output_organization", "metadata")
error_organization_directory = get_config_tag("error_organization", "metadata")
reuse_directory = get_config_tag("output_reuse", "metadata")
error_reuse_directory = get_config_tag("error_reuse", "metadata")

# parameters
reset = get_config_tag("reset", "metadata")

# collect metadata about datasets, organizations and reuses
collect_metadata.main(output_directory=output_directory,
                      organization_directory=organization_directory,
                      reuse_directory=reuse_directory,
                      error_directory=error_directory,
                      error_organization_directory=error_organization_directory,
                      error_reuse_directory=error_reuse_directory,
                      reset=reset)
