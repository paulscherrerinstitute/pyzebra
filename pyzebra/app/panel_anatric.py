import xml.etree.ElementTree as ET

from bokeh.layouts import column, row
from bokeh.models import Button, Panel, RadioButtonGroup, Select, Spinner, TextAreaInput, TextInput

import pyzebra


def create():
    def fileinput_callback(_attr, _old, new):
        tree = ET.parse(new)

        logfile_elem = tree.find("logfile")
        logfile_textinput.value = logfile_elem.attrib["file"]
        logfile_verbosity_select.value = logfile_elem.attrib["verbosity"]

        filelist_elem = tree.find("FileList")
        if filelist_elem is None:
            filelist_elem = tree.find("SinqFileList")
            filelist_type.value = "SINQ"
        else:
            filelist_type.value = "TRICS"

        filelist_format_textinput.value = filelist_elem.attrib["format"]
        filelist_datapath_textinput.value = filelist_elem.find("datapath").attrib["value"]
        range_vals = filelist_elem.find("range").attrib
        filelist_ranges_textareainput.value = str(
            (int(range_vals["start"]), int(range_vals["end"]))
        )

        crystal_elem = tree.find("crystal")
        crystal_sample_textinput.value = crystal_elem.find("Sample").attrib["name"]

        lambda_elem = crystal_elem.find("lambda")
        if lambda_elem is not None:
            lambda_textinput.value = lambda_elem.attrib["value"]

        zeroOM_elem = crystal_elem.find("zeroOM")
        if zeroOM_elem is not None:
            zeroOM_textinput.value = zeroOM_elem.attrib["value"]

        zeroSTT_elem = crystal_elem.find("zeroSTT")
        if zeroSTT_elem is not None:
            zeroSTT_textinput.value = zeroSTT_elem.attrib["value"]

        zeroCHI_elem = crystal_elem.find("zeroCHI")
        if zeroCHI_elem is not None:
            zeroCHI_textinput.value = zeroCHI_elem.attrib["value"]

        ub_textareainput.value = crystal_elem.find("UB").text

        dataFactory_elem = tree.find("DataFactory")
        dist1_textinput.value = dataFactory_elem.find("dist1").attrib["value"]

        reflectionPrinter_elem = tree.find("ReflectionPrinter")
        reflectionPrinter_format_select.value = reflectionPrinter_elem.attrib["format"]

        alg_elem = tree.find("Algorithm")
        if alg_elem.attrib["implementation"] == "adaptivemaxcog":
            set_active_widgets("adaptivemaxcog")

            threshold_spinner.value = float(alg_elem.find("threshold").attrib["value"])
            shell_spinner.value = float(alg_elem.find("shell").attrib["value"])
            steepness_spinner.value = float(alg_elem.find("steepness").attrib["value"])
            duplicateDistance_spinner.value = float(
                alg_elem.find("duplicateDistance").attrib["value"]
            )
            maxequal_spinner.value = float(alg_elem.find("maxequal").attrib["value"])
            # apd_window_spinner.value = float(alg_elem.find("window").attrib["value"])

        elif alg_elem.attrib["implementation"] == "adaptivedynamic":
            set_active_widgets("adaptivedynamic")

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

    def set_active_widgets(implementation):
        if implementation == "adaptivemaxcog":
            mode_radio_button_group.active = 0
            disable_adaptivemaxcog = False
            disable_adaptivedynamic = True

        elif implementation == "adaptivedynamic":
            mode_radio_button_group.active = 1
            disable_adaptivemaxcog = True
            disable_adaptivedynamic = False
        else:
            raise ValueError("Implementation can be either 'adaptivemaxcog' or 'adaptivedynamic'")

        threshold_spinner.disabled = disable_adaptivemaxcog
        shell_spinner.disabled = disable_adaptivemaxcog
        steepness_spinner.disabled = disable_adaptivemaxcog
        duplicateDistance_spinner.disabled = disable_adaptivemaxcog
        maxequal_spinner.disabled = disable_adaptivemaxcog
        apd_window_spinner.disabled = disable_adaptivemaxcog

        admi_window_spinner.disabled = disable_adaptivedynamic
        border_spinner.disabled = disable_adaptivedynamic
        minWindow_spinner.disabled = disable_adaptivedynamic
        reflectionFile_spinner.disabled = disable_adaptivedynamic
        targetMonitor_spinner.disabled = disable_adaptivedynamic
        smoothSize_spinner.disabled = disable_adaptivedynamic
        loop_spinner.disabled = disable_adaptivedynamic
        minPeakCount_spinner.disabled = disable_adaptivedynamic
        displacementCurve_spinner.disabled = disable_adaptivedynamic

    fileinput = TextInput(title="Path to XML configuration file:", width=600)
    fileinput.on_change("value", fileinput_callback)

    # General parameters
    # ---- logfile
    logfile_textinput = TextInput(title="Logfile:", value="logfile.log", width=520)
    logfile_verbosity_select = Select(
        title="verbosity:", options=["0", "5", "10", "15", "30"], width=70
    )

    # ---- FileList
    filelist_type = Select(title="File List:", options=["TRICS", "SINQ"], width=100)
    filelist_format_textinput = TextInput(title="format:", width=490)
    filelist_datapath_textinput = TextInput(title="datapath:")
    filelist_ranges_textareainput = TextAreaInput(title="ranges:", height=100)

    # ---- crystal
    crystal_sample_textinput = TextInput(title="Sample Name:")
    lambda_textinput = TextInput(title="lambda:", width=140)
    ub_textareainput = TextAreaInput(title="UB matrix:", height=100)
    zeroOM_textinput = TextInput(title="zeroOM:", width=140)
    zeroSTT_textinput = TextInput(title="zeroSTT:", width=140)
    zeroCHI_textinput = TextInput(title="zeroCHI:", width=140)

    # ---- DataFactory
    dist1_textinput = TextInput(title="Dist1:", width=290)

    # ---- BackgroundProcessor

    # ---- DetectorEfficency

    # ---- ReflectionPrinter
    reflectionPrinter_format_select = Select(
        title="ReflectionPrinter format:",
        options=[
            "rafin",
            "rafinf",
            "rafin2d",
            "rafin2di",
            "orient",
            "shelx",
            "jana2k",
            "jana2kf",
            "raw",
            "oksana",
        ],
        width=300,
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

    def mode_radio_button_group_callback(active):
        if active == 0:
            set_active_widgets("adaptivemaxcog")
        else:
            set_active_widgets("adaptivedynamic")

    mode_radio_button_group = RadioButtonGroup(
        labels=["Adaptive Peak Detection", "Adaptive Dynamic Mask Integration"], active=0
    )
    mode_radio_button_group.on_click(mode_radio_button_group_callback)
    set_active_widgets("adaptivemaxcog")

    def process_button_callback():
        pyzebra.anatric(fileinput.value)

    process_button = Button(label="Process", button_type="primary")
    process_button.on_click(process_button_callback)

    tab_layout = row(
        column(
            fileinput,
            row(logfile_textinput, logfile_verbosity_select),
            row(filelist_type, filelist_format_textinput),
            filelist_datapath_textinput,
            filelist_ranges_textareainput,
            crystal_sample_textinput,
            row(lambda_textinput, zeroOM_textinput, zeroSTT_textinput, zeroCHI_textinput),
            ub_textareainput,
            row(dist1_textinput, reflectionPrinter_format_select),
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
