import torch


"""from Simulations import tm, be
import torch
be = torch.tensor(be, dtype=torch.float32)
tm = torch.tensor(tm, dtype=torch.float32)


adc = torch.tensor([[0.6247],
        [0.6239],
        [0.6251],
        [0.6248],
        [0.6248],
        [0.6247],
        [0.6235],
        [0.6254],
        [0.6237],
        [0.6245],
        [0.6246],
        [0.6245],
        [0.6240],
        [0.6237],
        [0.6250],
        [0.6236],
        [0.6231],
        [0.6239],
        [0.6271],
        [0.6237],
        [0.6235],
        [0.6250],
        [0.6248],
        [0.6250],
        [0.6252],
        [0.6252],
        [0.6233],
        [0.6247],
        [0.6237],
        [0.6254],
        [0.6249],
        [0.6240],
        [0.6253],
        [0.6250],
        [0.6245],
        [0.6247],
        [0.6239],
        [0.6250],
        [0.6235],
        [0.6253],
        [0.6242],
        [0.6251],
        [0.6242],
        [0.6251],
        [0.6252],
        [0.6251],
        [0.6252],
        [0.6254],
        [0.6253],
        [0.6236],
        [0.6249],
        [0.6250],
        [0.6237],
        [0.6253],
        [0.6233],
        [0.6242],
        [0.6253],
        [0.6250],
        [0.6272],
        [0.6253],
        [0.6249],
        [0.6236],
        [0.6237],
        [0.6236],
        [0.6243],
        [0.6248],
        [0.6250],
        [0.6254],
        [0.6240],
        [0.6245],
        [0.6231],
        [0.6239],
        [0.6248],
        [0.6254],
        [0.6246],
        [0.6230],
        [0.6246],
        [0.6244],
        [0.6237],
        [0.6253],
        [0.6237],
        [0.6240],
        [0.6268],
        [0.6244],
        [0.6245],
        [0.6240],
        [0.6251],
        [0.6245],
        [0.6247],
        [0.6235],
        [0.6246],
        [0.6253],
        [0.6252],
        [0.6242],
        [0.6241],
        [0.6239],
        [0.6247],
        [0.6249],
        [0.6236],
        [0.6250],
        [0.6249],
        [0.6236],
        [0.6254],
        [0.6252],
        [0.6242],
        [0.6244],
        [0.6239],
        [0.6253],
        [0.6246],
        [0.6254],
        [0.6245],
        [0.6237],
        [0.6248],
        [0.6254],
        [0.6229],
        [0.6235],
        [0.6239],
        [0.6240],
        [0.6252],
        [0.6235],
        [0.6251],
        [0.6243],
        [0.6235],
        [0.6241],
        [0.6248],
        [0.6241],
        [0.6255],
        [0.6248]])
sigma = torch.tensor([[0.5172],
        [0.5174],
        [0.5169],
        [0.5171],
        [0.5172],
        [0.5178],
        [0.5177],
        [0.5167],
        [0.5178],
        [0.5173],
        [0.5176],
        [0.5174],
        [0.5176],
        [0.5176],
        [0.5169],
        [0.5176],
        [0.5178],
        [0.5178],
        [0.5172],
        [0.5175],
        [0.5178],
        [0.5173],
        [0.5171],
        [0.5169],
        [0.5168],
        [0.5168],
        [0.5177],
        [0.5171],
        [0.5175],
        [0.5167],
        [0.5171],
        [0.5174],
        [0.5167],
        [0.5171],
        [0.5177],
        [0.5172],
        [0.5178],
        [0.5170],
        [0.5176],
        [0.5167],
        [0.5173],
        [0.5177],
        [0.5172],
        [0.5169],
        [0.5168],
        [0.5169],
        [0.5168],
        [0.5174],
        [0.5167],
        [0.5176],
        [0.5173],
        [0.5171],
        [0.5179],
        [0.5167],
        [0.5179],
        [0.5172],
        [0.5167],
        [0.5175],
        [0.5173],
        [0.5171],
        [0.5169],
        [0.5180],
        [0.5178],
        [0.5179],
        [0.5177],
        [0.5170],
        [0.5172],
        [0.5170],
        [0.5173],
        [0.5172],
        [0.5179],
        [0.5175],
        [0.5170],
        [0.5167],
        [0.5174],
        [0.5179],
        [0.5176],
        [0.5175],
        [0.5177],
        [0.5167],
        [0.5177],
        [0.5176],
        [0.5174],
        [0.5178],
        [0.5177],
        [0.5175],
        [0.5168],
        [0.5172],
        [0.5174],
        [0.5178],
        [0.5170],
        [0.5168],
        [0.5168],
        [0.5173],
        [0.5175],
        [0.5178],
        [0.5172],
        [0.5169],
        [0.5179],
        [0.5173],
        [0.5169],
        [0.5175],
        [0.5167],
        [0.5168],
        [0.5178],
        [0.5175],
        [0.5176],
        [0.5167],
        [0.5171],
        [0.5167],
        [0.5173],
        [0.5179],
        [0.5170],
        [0.5170],
        [0.5179],
        [0.5179],
        [0.5179],
        [0.5180],
        [0.5168],
        [0.5178],
        [0.5168],
        [0.5174],
        [0.5178],
        [0.5177],
        [0.5178],
        [0.5178],
        [0.5167],
        [0.5175]])
axr = torch.tensor([[0.7577],
        [0.7586],
        [0.7572],
        [0.7575],
        [0.7576],
        [0.7583],
        [0.7591],
        [0.7567],
        [0.7592],
        [0.7579],
        [0.7584],
        [0.7581],
        [0.7586],
        [0.7588],
        [0.7572],
        [0.7589],
        [0.7595],
        [0.7588],
        [0.7568],
        [0.7587],
        [0.7592],
        [0.7578],
        [0.7575],
        [0.7572],
        [0.7570],
        [0.7570],
        [0.7592],
        [0.7576],
        [0.7588],
        [0.7568],
        [0.7574],
        [0.7584],
        [0.7569],
        [0.7575],
        [0.7582],
        [0.7577],
        [0.7591],
        [0.7573],
        [0.7589],
        [0.7568],
        [0.7581],
        [0.7583],
        [0.7581],
        [0.7571],
        [0.7570],
        [0.7571],
        [0.7569],
        [0.7577],
        [0.7568],
        [0.7589],
        [0.7579],
        [0.7574],
        [0.7591],
        [0.7568],
        [0.7595],
        [0.7581],
        [0.7568],
        [0.7578],
        [0.7570],
        [0.7575],
        [0.7572],
        [0.7592],
        [0.7593],
        [0.7592],
        [0.7584],
        [0.7574],
        [0.7577],
        [0.7571],
        [0.7584],
        [0.7579],
        [0.7596],
        [0.7587],
        [0.7575],
        [0.7567],
        [0.7580],
        [0.7596],
        [0.7582],
        [0.7583],
        [0.7589],
        [0.7568],
        [0.7590],
        [0.7586],
        [0.7572],
        [0.7585],
        [0.7585],
        [0.7586],
        [0.7570],
        [0.7580],
        [0.7579],
        [0.7592],
        [0.7576],
        [0.7568],
        [0.7569],
        [0.7582],
        [0.7584],
        [0.7588],
        [0.7577],
        [0.7573],
        [0.7592],
        [0.7576],
        [0.7573],
        [0.7588],
        [0.7567],
        [0.7570],
        [0.7587],
        [0.7582],
        [0.7588],
        [0.7568],
        [0.7577],
        [0.7568],
        [0.7580],
        [0.7593],
        [0.7576],
        [0.7572],
        [0.7598],
        [0.7593],
        [0.7590],
        [0.7590],
        [0.7570],
        [0.7592],
        [0.7571],
        [0.7582],
        [0.7592],
        [0.7587],
        [0.7586],
        [0.7590],
        [0.7567],
        [0.7579]])


adc_prime = adc * (1 - sigma * torch.exp(- tm * axr))
E_vox = torch.exp(- adc_prime * be)
print("adc_prime:",adc_prime)
print("evox:",E_vox)

if torch.isnan(E_vox).any():
    print("evox nan")
else:
    print("evox good")
if torch.isnan(adc_prime).any():
    print("adc prime nan")
else:
    print("adc prime good")"""