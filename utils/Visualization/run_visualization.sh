python testRunner_tiled_siracusa_w_redmule.py -t Tests/testTrainCCT/CCT2/CCT2_LA2_lora  --defaultMemLevel L3 --cores 8 --l1 144000 --doublebuffer --plotMemAlloc --profileTiling \
  > /app/reports/CCT2_LA2_lora/latency.txt

python /app/reports/report2csv_summary.py /app/reports/CCT2_LA2_lora/latency.txt /app/reports/CCT2_LA2_lora/latency_report.csv > /app/reports/CCT2_LA2_lora/latency_report_summary.txt

python /app/Onnx4Deeploy/utils/htmlanalyzer.py /app/reports/CCT2_LA2_lora/memory_alloc.html > /app/reports/CCT2_LA2_lora/memory_alloc.txt