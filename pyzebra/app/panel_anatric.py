import xml.etree.ElementTree as ET

from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    Div,
    Panel,
    RadioButtonGroup,
    RangeSlider,
    Select,
    Spinner,
    TextInput,
)

import pyzebra


def create():
    def fileinput_callback(_attr, _old, new):
        tree = ET.parse(new)

        logfile_elem = tree.find("logfile")
        logfile_textinput.value = logfile_elem.attrib["file"]
        logfile_verbosity_select.value = logfile_elem.attrib["verbosity"]

        filelist_elem = tree.find("FileList")
        filelist_format_textinput.value = filelist_elem.attrib["format"]
        filelist_datapath_textinput.value = filelist_elem.find("datapath").attrib["value"]
        range_vals = filelist_elem.find("range").attrib
        filelist_range_rangeslider.value = (int(range_vals["start"]), int(range_vals["end"]))

        alg_elem = tree.find("Algorithm")
        if alg_elem.attrib["implementation"] == "adaptivemaxcog":
            mode_radio_button_group.active = 0

            threshold_spinner.value = float(alg_elem.find("threshold").attrib["value"])
            shell_spinner.value = float(alg_elem.find("shell").attrib["value"])
            steepness_spinner.value = float(alg_elem.find("steepness").attrib["value"])
            duplicateDistance_spinner.value = float(
                alg_elem.find("duplicateDistance").attrib["value"]
            )
            maxequal_spinner.value = float(alg_elem.find("maxequal").attrib["value"])
            # apd_window_spinner.value = float(alg_elem.find("window").attrib["value"])

        elif alg_elem.attrib["implementation"] == "adaptivedynamic":
            mode_radio_button_group.active = 1

            # admi_window_spinner.value = float(alg_elem.find("window").attrib["value"])
            # .value = float(alg_elem.find("border").attrib["value"])
            # minWindow_spinner.value = float(alg_elem.find("minWindow").attrib["value"])
            # reflectionFile_spinner.value = float(alg_elem.find("reflectionFile").attrib["value"])
            targetMonitor_spinner.value = float(alg_elem.find("targetMonitor").attrib["value"])
            smoothSize_spinner.value = float(alg_elem.find("smoothSize").attrib["value"])
            loop_spinner.value = float(alg_elem.find("loop").attrib["value"])
            minPeakCount_spinner.value = float(alg_elem.find("minPeakCount").attrib["value"])
            # displacementCurve_spinner.value = float(alg_elem.find("threshold").attrib["value"])
        else:
            raise ValueError("Unknown processing mode.")

    fileinput = TextInput(title="Path to XML configuration file:", width=600)
    fileinput.on_change("value", fileinput_callback)

    # General parameters
    # ---- logfile
    logfile_textinput = TextInput(title="Logfile:", value="logfile.log", width=520)
    logfile_verbosity_select = Select(
        title="verbosity:", options=["0", "5", "10", "15", "30"], width=70
    )

    # ---- FileList
    filelist_div = Div(text="File List:", width=100)
    filelist_format_textinput = TextInput(title="format")
    filelist_datapath_textinput = TextInput(title="datapath")
    filelist_range_rangeslider = RangeSlider(title="range", start=0, end=2000, value=(0, 2000))

    # ---- crystal

    # ---- DataFactory

    # ---- BackgroundProcessor

    # ---- DetectorEfficency

    # ---- ReflectionPrinter

    mode_radio_button_group = RadioButtonGroup(
        labels=["Adaptive Peak Detection", "Adaptive Dynamic Mask Integration"], active=0
    )

    # Adaptive Peak Detection (adaptivemaxcog)
    # ---- threshold
    threshold_spinner = Spinner(title="Threshold", value=None)

    # ---- shell
    shell_spinner = Spinner(title="Shell", value=None, low=0)

    # ---- steepness
    steepness_spinner = Spinner(title="Steepness", value=None)

    # ---- duplicateDistance
    duplicateDistance_spinner = Spinner(title="Duplicate Distance", value=None)

    # ---- maxequal
    maxequal_spinner = Spinner(title="Max Equal", value=None)

    # ---- window
    apd_window_spinner = Spinner(title="Window", value=None)

    # Adaptive Dynamic Mask Integration (adaptivedynamic)
    # ---- window
    admi_window_spinner = Spinner(title="Window", value=None)

    # ---- border
    border_spinner = Spinner(title="Border", value=None)

    # ---- minWindow
    minWindow_spinner = Spinner(title="Min Window", value=None)

    # ---- reflectionFile
    reflectionFile_spinner = Spinner(title="Reflection File", value=None)

    # ---- targetMonitor
    targetMonitor_spinner = Spinner(title="Target Monitor", value=None)

    # ---- smoothSize
    smoothSize_spinner = Spinner(title="Smooth Size", value=None)

    # ---- loop
    loop_spinner = Spinner(title="Loop", value=None)

    # ---- minPeakCount
    minPeakCount_spinner = Spinner(title="Min Peak Count", value=None)

    # ---- displacementCurve
    displacementCurve_spinner = Spinner(title="Displacement Curve", value=None)

    def process_button_callback():
        pyzebra.anatric(fileinput.value)

    process_button = Button(label="Process")
    process_button.on_click(process_button_callback)

    tab_layout = row(
        column(
            fileinput,
            row(logfile_textinput, logfile_verbosity_select),
            filelist_div,
            filelist_format_textinput,
            filelist_datapath_textinput,
            filelist_range_rangeslider,
            process_button,
        ),
        column(
            mode_radio_button_group,
            row(
                column(
                    threshold_spinner,
                    shell_spinner,
                    steepness_spinner,
                    duplicateDistance_spinner,
                    maxequal_spinner,
                    apd_window_spinner,
                ),
                column(
                    admi_window_spinner,
                    border_spinner,
                    minWindow_spinner,
                    reflectionFile_spinner,
                    targetMonitor_spinner,
                    smoothSize_spinner,
                    loop_spinner,
                    minPeakCount_spinner,
                    displacementCurve_spinner,
                ),
            ),
        ),
    )

    return Panel(child=tab_layout, title="Anatric")
