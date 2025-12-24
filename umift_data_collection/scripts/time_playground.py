#%%
from umift.utils.time_utils import ntp_time_to_timestamp_direct, convert_timestamp_to_iso_processed, isostringformat_to_timestamp, ntp_time_to_timestamp

time_from_arkit = "2024-11-19T21:12:06.417Z"

# NTP time
ntp_coinFT_time = "1.730769830135123968e+09"

iso_ntp_time=convert_timestamp_to_iso_processed(ntp_coinFT_time)
print('iso_ntp_time: ', iso_ntp_time)
timestamp_back_npt_time = isostringformat_to_timestamp(iso_ntp_time)
print('timestamp_back_npt_time: ', timestamp_back_npt_time)

timestamp = ntp_time_to_timestamp(ntp_coinFT_time)
direct_timestamp = ntp_time_to_timestamp_direct(ntp_coinFT_time)

print('timestamp: ', timestamp)
print('direct_timestamp: ', direct_timestamp)

# %%
