from bokeh.layouts import column, row
from bokeh.models import Button, Panel, RadioButtonGroup, Select, TextAreaInput, TextInput

import pyzebra


def create():
    config = pyzebra.AnatricConfig()

    def fileinput_callback(_attr, _old, new):
        config.load_from_file(new)

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
            # aps_window_textinput.value = str(tuple(config.aps_window.values()))

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
    def logfile_textinput_callback(_attr, _old, new):
        config.logfile = new

    logfile_textinput = TextInput(title="Logfile:", value="logfile.log", width=520)
    logfile_textinput.on_change("value", logfile_textinput_callback)

    def logfile_verbosity_select_callback(_attr, _old, new):
        config.logfile_verbosity = new

    logfile_verbosity_select = Select(
        title="verbosity:", options=["0", "5", "10", "15", "30"], width=70
    )
    logfile_verbosity_select.on_change("value", logfile_verbosity_select_callback)

    # ---- FileList
    def filelist_type_callback(_attr, _old, new):
        config.filelist_type = new

    filelist_type = Select(title="File List:", options=["TRICS", "SINQ"], width=100)
    filelist_type.on_change("value", filelist_type_callback)

    def filelist_format_textinput_callback(_attr, _old, new):
        config.filelist_format = new

    filelist_format_textinput = TextInput(title="format:", width=490)
    filelist_format_textinput.on_change("value", filelist_format_textinput_callback)

    def filelist_datapath_textinput_callback(_attr, _old, new):
        config.filelist_datapath = new

    filelist_datapath_textinput = TextInput(title="datapath:")
    filelist_datapath_textinput.on_change("value", filelist_datapath_textinput_callback)

    def filelist_ranges_textareainput_callback(_attr, _old, new):
        config.ranges = new

    filelist_ranges_textareainput = TextAreaInput(title="ranges:", height=100)
    filelist_ranges_textareainput.on_change("value", filelist_ranges_textareainput_callback)

    # ---- crystal
    def crystal_sample_textinput_callback(_attr, _old, new):
        config.crystal_sample = new

    crystal_sample_textinput = TextInput(title="Sample Name:")
    crystal_sample_textinput.on_change("value", crystal_sample_textinput_callback)

    def lambda_textinput_callback(_attr, _old, new):
        config.crystal_lambda = new

    lambda_textinput = TextInput(title="lambda:", width=140)
    lambda_textinput.on_change("value", lambda_textinput_callback)

    def ub_textareainput_callback(_attr, _old, new):
        config.crystal_UB = new

    ub_textareainput = TextAreaInput(title="UB matrix:", height=100)
    ub_textareainput.on_change("value", ub_textareainput_callback)

    def zeroOM_textinput_callback(_attr, _old, new):
        config.crystal_zeroOM = new

    zeroOM_textinput = TextInput(title="zeroOM:", width=140)
    zeroOM_textinput.on_change("value", zeroOM_textinput_callback)

    def zeroSTT_textinput_callback(_attr, _old, new):
        config.crystal_zeroSTT = new

    zeroSTT_textinput = TextInput(title="zeroSTT:", width=140)
    zeroSTT_textinput.on_change("value", zeroSTT_textinput_callback)

    def zeroCHI_textinput_callback(_attr, _old, new):
        config.crystal_zeroCHI = new

    zeroCHI_textinput = TextInput(title="zeroCHI:", width=140)
    zeroCHI_textinput.on_change("value", zeroCHI_textinput_callback)

    # ---- DataFactory
    def dist1_textinput_callback(_attr, _old, new):
        config.dist1 = new

    dist1_textinput = TextInput(title="Dist1:", width=290)
    dist1_textinput.on_change("value", dist1_textinput_callback)

    # ---- BackgroundProcessor

    # ---- DetectorEfficency

    # ---- ReflectionPrinter
    def reflectionPrinter_format_select_callback(_attr, _old, new):
        config.reflectionPrinter_format = new

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
    reflectionPrinter_format_select.on_change("value", reflectionPrinter_format_select_callback)

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
