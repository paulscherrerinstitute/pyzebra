from bokeh.layouts import column, row
from bokeh.models import Button, Panel, RadioButtonGroup, Select, Spinner, TextAreaInput, TextInput

import pyzebra


def create():
    def fileinput_callback(_attr, _old, new):
        config = pyzebra.AnatricConfig(new)

        logfile_textinput.value = config.logfile
        logfile_verbosity_select.value = config.logfile_verbosity

        filelist_type.value = config.filelist_type
        filelist_format_textinput.value = config.filelist_format
        filelist_datapath_textinput.value = config.filelist_datapath
        filelist_ranges_textareainput.value = str(config.filelist_ranges)

        crystal_sample_textinput.value = config.crystal_sample
        lambda_textinput.value = config.crystal_lambda
        zeroOM_textinput.value = config.crystal_zeroOM
        zeroSTT_textinput.value = config.crystal_zeroSTT
        zeroCHI_textinput.value = config.crystal_zeroCHI
        ub_textareainput.value = config.crystal_UB

        dist1_textinput.value = config.dist1
        reflectionPrinter_format_select.value = config.reflectionPrinter_format

        set_active_widgets(config.algorithm)
        if config.algorithm == "adaptivemaxcog":
            threshold_spinner.value = float(config.threshold)
            shell_spinner.value = float(config.shell)
            steepness_spinner.value = float(config.steepness)
            duplicateDistance_spinner.value = float(config.duplicateDistance)
            maxequal_spinner.value = float(config.maxequal)
            # apd_window_spinner.value = float(config.apd_window)

        elif config.algorithm == "adaptivedynamic":
            # admi_window_spinner.value = float(config.admi_window)
            # border_spinner.value = float(config.border)
            # minWindow_spinner.value = float(config.minWindow)
            # reflectionFile_spinner.value = float(config.reflectionFile)
            targetMonitor_spinner.value = float(config.targetMonitor)
            smoothSize_spinner.value = float(config.smoothSize)
            loop_spinner.value = float(config.loop)
            minPeakCount_spinner.value = float(config.minPeakCount)
            # displacementCurve_spinner.value = float(config.displacementCurve)
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
