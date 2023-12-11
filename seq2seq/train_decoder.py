from decoder import Decoder
from character_loader import data_loader
import lightning as L

model = Decoder(95, 60, 30, 15)

trainer = L.Trainer(max_epochs=34, accelerator="auto", devices="auto")

trainer.fit(model, data_loader)


#%%
