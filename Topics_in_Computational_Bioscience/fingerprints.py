

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
# rdDepictor.SetPreferCoordGen(True)
from IPython.display import SVG

m1 = Chem.MolFromSmiles("CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5")
m2 = Chem.MolFromSmiles("CCC1=C2N=C(C=C(N2N=C1)NCC3=C[N+](=CC=C3)[O-])N4CCCC[C@H]4CCO")

d2d = rdMolDraw2D.MolDraw2DSVG(600,280,300,280)
d2d.DrawMolecules([m1,m2], legends=['a', 'b'])
d2d.FinishDrawing()
SVG(d2d.GetDrawingText())