from zipline.gens.transform import BatchWindow, batch_transform

class CovEventWindow(BatchWindow):
    def get_value(self, data):
        return data.cov()

@batch_transform
def cov(data):
    return data.cov()
