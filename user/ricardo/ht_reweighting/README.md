### HT reweighting

The low statistics of the DY NLO samples and the large amount of negative weights make the training of the DCR estimator with DY quite challenging. For this reason, it was decided to instead use the DY LO HT-binned samples, which have much larger statistics and (likely) a negligible fraction of negative weights.

This sample has a known mismodelling in high-pT jets from ISR (only possible source of jets in a DY). The [four-top analysis](https://cms.cern.ch/iCMS/analysisadmin/cadilines?line=TOP-24-008) employed a data-driven correction based on HT to correct for this mismodelling. Their corrections are not directly usable in our sample, as they were only derived for 4+ jet events, while our phase-space has a much lower jet multiplicity. The code in this repository is a minimal implementation of the code to derive that correction.

