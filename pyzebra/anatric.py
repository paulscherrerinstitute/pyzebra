import subprocess
import xml.etree.ElementTree as ET


def anatric(config_file):
    subprocess.run(["anatric", config_file], check=True)


class AnatricConfig:
    def __init__(self, filename=None):
        if filename:
            self.load_from_file(filename)

    def load_from_file(self, filename):
        tree = ET.parse(filename)
        self._tree = tree

        filelist_elem = tree.find("FileList")
        if filelist_elem is None:
            filelist_elem = tree.find("SinqFileList")
            self.filelist_type = "SINQ"
        else:
            self.filelist_type = "TRICS"

        self.filelist_format = filelist_elem.attrib["format"]
        self.filelist_datapath = filelist_elem.find("datapath").attrib["value"]
        range_vals = filelist_elem.find("range").attrib
        self.filelist_ranges = (int(range_vals["start"]), int(range_vals["end"]))

        crystal_elem = tree.find("crystal")
        self.crystal_sample = crystal_elem.find("Sample").attrib["name"]

        lambda_elem = crystal_elem.find("lambda")
        if lambda_elem is not None:
            self.crystal_lambda = lambda_elem.attrib["value"]
        else:
            self.crystal_lambda = None

        zeroOM_elem = crystal_elem.find("zeroOM")
        if zeroOM_elem is not None:
            self.crystal_zeroOM = zeroOM_elem.attrib["value"]
        else:
            self.crystal_zeroOM = None

        zeroSTT_elem = crystal_elem.find("zeroSTT")
        if zeroSTT_elem is not None:
            self.crystal_zeroSTT = zeroSTT_elem.attrib["value"]
        else:
            self.crystal_zeroSTT = None

        zeroCHI_elem = crystal_elem.find("zeroCHI")
        if zeroCHI_elem is not None:
            self.crystal_zeroCHI = zeroCHI_elem.attrib["value"]
        else:
            self.crystal_zeroCHI = None

        self.crystal_UB = crystal_elem.find("UB").text

        dataFactory_elem = tree.find("DataFactory")
        self.dist1 = dataFactory_elem.find("dist1").attrib["value"]

        reflectionPrinter_elem = tree.find("ReflectionPrinter")
        self.reflectionPrinter_format = reflectionPrinter_elem.attrib["format"]

        alg_elem = tree.find("Algorithm")
        self.algorithm = alg_elem.attrib["implementation"]
        if self.algorithm == "adaptivemaxcog":
            self.threshold = float(alg_elem.find("threshold").attrib["value"])
            self.shell = float(alg_elem.find("shell").attrib["value"])
            self.steepness = float(alg_elem.find("steepness").attrib["value"])
            self.duplicateDistance = float(alg_elem.find("duplicateDistance").attrib["value"])
            self.maxequal = float(alg_elem.find("maxequal").attrib["value"])
            # self.apd_window = float(alg_elem.find("window").attrib["value"])

        elif self.algorithm == "adaptivedynamic":
            # self.admi_window = float(alg_elem.find("window").attrib["value"])
            # self.border = float(alg_elem.find("border").attrib["value"])
            # self.minWindow = float(alg_elem.find("minWindow").attrib["value"])
            # self.reflectionFile = float(alg_elem.find("reflectionFile").attrib["value"])
            self.targetMonitor = float(alg_elem.find("targetMonitor").attrib["value"])
            self.smoothSize = float(alg_elem.find("smoothSize").attrib["value"])
            self.loop = float(alg_elem.find("loop").attrib["value"])
            self.minPeakCount = float(alg_elem.find("minPeakCount").attrib["value"])
            # self.displacementCurve = float(alg_elem.find("threshold").attrib["value"])
        else:
            raise ValueError("Unknown processing mode.")

    @property
    def logfile(self):
        return self._tree.find("logfile").attrib["file"]

    @logfile.setter
    def logfile(self, value):
        self._tree.find("logfile").attrib["file"] = value

    @property
    def logfile_verbosity(self):
        return self._tree.find("logfile").attrib["verbosity"]

    @logfile_verbosity.setter
    def logfile_verbosity(self, value):
        self._tree.find("logfile").attrib["verbosity"] = value

    def save_as(self, filename):
        self._tree.write(filename)
