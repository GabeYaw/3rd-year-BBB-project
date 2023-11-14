### lr = 0.01
100,000 voxels 67 epochs then error. Python was using 60Gb of memory. See screenshot on desktop

90.000 voxels 82 epochs but only got to fourth breakpoint, but code did say 'done', so not an error

80,000 voxels 118 epochs no error

50,000 voxels 59 epochs no error message, but long delay, doesn't seem to reach past 5th breakpoint

50,000 voxels attempt 2: 36 epochs

lr = 0.0001 

### 50,000 voxels 101 epochs, only got up to fourth breakpoint


In general the last section of the code after breakpoint 5 seems to use up a lot of memory, in the tens of Gb