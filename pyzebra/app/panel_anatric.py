from bokeh.layouts import column, row
from bokeh.models import Button, Panel, RadioButtonGroup, Select, TextAreaInput, TextInput

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
            threshold_textinput.value = config.threshold
            shell_textinput.value = config.shell
            steepness_textinput.value = config.steepness
            duplicateDistance_textinput.value = config.duplicateDistance
            maxequal_textinput.value = config.maxequal
            # aps_window_textinput.value = config.aps_window

        elif config.algorithm == "adaptivedynamic":
            # adm_window_textinput.value = config.adm_window
            # border_textinput.value = config.border
            # minWindow_textinput.value = config.minWindow
            reflectionFile_textinput.value = config.reflectionFile
            targetMonitor_textinput.value = config.targetMonitor
            smoothSize_textinput.value = config.smoothSize
            loop_textinput.value = config.loop
            minPeakCount_textinput.value = config.minPeakCount
            # displacementCurve_textinput.value = config.displacementCurve
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

        threshold_textinput.disabled = disable_adaptivemaxcog
        shell_textinput.disabled = disable_adaptivemaxcog
        steepness_textinput.disabled = disable_adaptivemaxcog
        duplicateDistance_textinput.disabled = disable_adaptivemaxcog
        maxequal_textinput.disabled = disable_adaptivemaxcog
        aps_window_textinput.disabled = disable_adaptivemaxcog

        adm_window_textinput.disabled = disable_adaptivedynamic
        border_textinput.disabled = disable_adaptivedynamic
        minWindow_textinput.disabled = disable_adaptivedynamic
        reflectionFile_textinput.disabled = disable_adaptivedynamic
        targetMonitor_textinput.disabled = disable_adaptivedynamic
        smoothSize_textinput.disabled = disable_adaptivedynamic
        loop_textinput.disabled = disable_adaptivedynamic
        minPeakCount_textinput.disabled = disable_adaptivedynamic
        displacementCurve_textinput.disabled = disable_adaptivedynamic

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
    threshold_textinput = TextInput(title="Threshold")

    # ---- shell
    shell_textinput = TextInput(title="Shell")

    # ---- steepness
    steepness_textinput = TextInput(title="Steepness")

    # ---- duplicateDistance
    duplicateDistance_textinput = TextInput(title="Duplicate Distance")

    # ---- maxequal
    maxequal_textinput = TextInput(title="Max Equal")

    # ---- window
    aps_window_textinput = TextInput(title="Window")

    # Adaptive Dynamic Mask Integration (adaptivedynamic)
    # ---- window
    adm_window_textinput = TextInput(title="Window")

    # ---- border
    border_textinput = TextInput(title="Border")

    # ---- minWindow
    minWindow_textinput = TextInput(title="Min Window")

    # ---- reflectionFile
    reflectionFile_textinput = TextInput(title="Reflection File")

    # ---- targetMonitor
    targetMonitor_textinput = TextInput(title="Target Monitor")

    # ---- smoothSize
    smoothSize_textinput = TextInput(title="Smooth Size")

    # ---- loop
    loop_textinput = TextInput(title="Loop")

    # ---- minPeakCount
    minPeakCount_textinput = TextInput(title="Min Peak Count")

    # ---- displacementCurve
    displacementCurve_textinput = TextInput(title="Displacement Curve")

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
                    threshold_textinput,
                    shell_textinput,
                    steepness_textinput,
                    duplicateDistance_textinput,
                    maxequal_textinput,
                    aps_window_textinput,
                ),
                column(
                    adm_window_textinput,
                    border_textinput,
                    minWindow_textinput,
                    reflectionFile_textinput,
                    targetMonitor_textinput,
                    smoothSize_textinput,
                    loop_textinput,
                    minPeakCount_textinput,
                    displacementCurve_textinput,
                ),
            ),
        ),
    )

    return Panel(child=tab_layout, title="Anatric")
