import lhapdf
import eko
import pathlib
from ekobox.evol_pdf import evolve_pdfs
from ekobox.cards import example
import toy_pdf as toy

pdf = toy.mkPDF("", 0)
th_card = example.theory()
op_card = example.operator()
# here we replace the grid with a very minimal one, to speed up the example
op_card.xgrid = eko.interpolation.XGrid([1e-3, 1e-2, 1e-1, 5e-1, 1.0])
op_card.mugrid = [(10.0, 5), (100.0, 5)]
# set QCD LO evolution
th_card.orders = (1, 0)

path = pathlib.Path("./myeko2.tar")
evolve_pdfs([pdf], th_card, op_card, install=True, name="Evolved_PDF", store_path=path)

evolved_pdf = lhapdf.mkPDF("Evolved_PDF", 0)

pid = 21  # gluon pid
Q2 = 89.10  #  Q^2 in Gev^2
x = 0.01  # momentum fraction

# check that the particle is present
print("has gluon?", evolved_pdf.hasFlavor(pid))

xg = evolved_pdf.xfxQ2(pid, x, Q2)
print(f"xg(x={x}, Q2={Q2}) = {xg}")
