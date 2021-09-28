# ir-annotation-extraction
Extracts annotations from a document scanned under infrared light and inserts those annotations into the original PDF.

# Usage

`python3 insert_annotation.py navigation_1.pdf transparent_navigation.png navigation_ir.jpg bias.png`

# Notes

## Timings (in seconds)

read: 0.4200859069824219
bb: 0.10314464569091797
homo: 0.0001049041748046875
warp: 0.03565788269042969
write: 0.12440276145935059
annot: 3.0618648529052734

# References

Test image taken from the paper *[The interplay of pedestrian navigation, wayfinding devices, and environmental features in indoor settings](https://dl.acm.org/doi/abs/10.1145/2857491.2857533)*.

Verena Schnitzler, Ioannis Giannopoulos, Christoph Hölscher, and Iva Barisic. 2016. The interplay of pedestrian navigation, wayfinding devices, and environmental features in indoor settings. In Proceedings of the Ninth Biennial ACM Symposium on Eye Tracking Research & Applications (ETRA '16). Association for Computing Machinery, New York, NY, USA, 85–93. DOI:https://doi.org/10.1145/2857491.2857533
