# -*- coding: utf-8 -*-
from dku_plugin_test_utils import dss_scenario

TEST_PROJECT_KEY = "PLUGINTESTMODELERRORANALYSIS"

def test_run_parser_tests(user_dss_clients):
    dss_scenario.run(user_dss_clients, project_key=TEST_PROJECT_KEY, scenario_id="PARSERTESTS")

def test_run_visualizer_tests(user_dss_clients):
    dss_scenario.run(user_dss_clients, project_key=TEST_PROJECT_KEY, scenario_id="VIZTESTS")
