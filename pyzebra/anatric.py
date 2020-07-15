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

    @property
    def filelist_type(self):
        if self._tree.find("FileList") is not None:
            return "TRICS"
        return "SINQ"

    @filelist_type.setter
    def filelist_type(self, value):
        if value == "TRICS":
            tag = "FileList"
        elif value == "SINQ":
            tag = "SinqFileList"
        else:
            raise ValueError("FileList value can only by 'TRICS' or 'SINQ'")

        self._tree.find("FileList").tag = tag

    @property
    def filelist_format(self):
        if self.filelist_type == "TRICS":
            return self._tree.find("FileList").attrib["format"]
        return self._tree.find("SinqFileList").attrib["format"]

    @filelist_format.setter
    def filelist_format(self, value):
        if self.filelist_type == "TRICS":
            self._tree.find("FileList").attrib["format"] = value
        else:  # SINQ
            self._tree.find("SinqFileList").attrib["format"] = value

    @property
    def filelist_datapath(self):
        if self.filelist_type == "TRICS":
            return self._tree.find("FileList").find("datapath").attrib["value"]
        return self._tree.find("SinqFileList").find("datapath").attrib["value"]

    @filelist_datapath.setter
    def filelist_datapath(self, value):
        if self.filelist_type == "TRICS":
            self._tree.find("FileList").find("datapath").attrib["value"] = value
        else:  # SINQ
            self._tree.find("SinqFileList").find("datapath").attrib["value"] = value

    @property
    def filelist_ranges(self):
        if self.filelist_type == "TRICS":
            range_vals = self._tree.find("FileList").find("range").attrib
        else:  # SINQ
            range_vals = self._tree.find("SinqFileList").find("range").attrib

        return (int(range_vals["start"]), int(range_vals["end"]))

    @filelist_ranges.setter
    def filelist_ranges(self, value):
        if self.filelist_type == "TRICS":
            range_vals = self._tree.find("FileList").find("range").attrib
        else:  # SINQ
            range_vals = self._tree.find("SinqFileList").find("range").attrib

        range_vals["start"] = str(value[0])
        range_vals["end"] = str(value[1])

    @property
    def crystal_sample(self):
        return self._tree.find("crystal").find("Sample").attrib["name"]

    @crystal_sample.setter
    def crystal_sample(self, value):
        self._tree.find("crystal").find("Sample").attrib["name"] = value

    @property
    def crystal_lambda(self):
        elem = self._tree.find("crystal").find("lambda")
        if elem is not None:
            return elem.attrib["value"]
        return None

    @crystal_lambda.setter
    def crystal_lambda(self, value):
        self._tree.find("crystal").find("lambda").attrib["value"] = value

    @property
    def crystal_zeroOM(self):
        elem = self._tree.find("crystal").find("zeroOM")
        if elem is not None:
            return elem.attrib["value"]
        return None

    @crystal_zeroOM.setter
    def crystal_zeroOM(self, value):
        self._tree.find("crystal").find("zeroOM").attrib["value"] = value

    @property
    def crystal_zeroSTT(self):
        elem = self._tree.find("crystal").find("zeroSTT")
        if elem is not None:
            return elem.attrib["value"]
        return None

    @crystal_zeroSTT.setter
    def crystal_zeroSTT(self, value):
        self._tree.find("crystal").find("zeroSTT").attrib["value"] = value

    @property
    def crystal_zeroCHI(self):
        elem = self._tree.find("crystal").find("zeroCHI")
        if elem is not None:
            return elem.attrib["value"]
        return None

    @crystal_zeroCHI.setter
    def crystal_zeroCHI(self, value):
        self._tree.find("crystal").find("zeroCHI").attrib["value"] = value

    @property
    def crystal_UB(self):
        elem = self._tree.find("crystal").find("UB")
        if elem is not None:
            return elem.text
        return None

    @crystal_UB.setter
    def crystal_UB(self, value):
        self._tree.find("crystal").find("UB").text = value

    def save_as(self, filename):
        self._tree.write(filename)
