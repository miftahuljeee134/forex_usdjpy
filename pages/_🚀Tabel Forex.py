import streamlit as st
import pandas as pd
import yfinance as yf

def get_forex_data(pairs):
    # Menyiapkan data untuk ditampilkan dalam tabel
    forex_list = []
    for pair in pairs:
        forex_info = yf.Ticker(pair)
        if forex_info:
            forex_name = forex_info.info.get('longName', pair)
            forex_symbol = pair
            
            # Mendapatkan harga forex
            forex_price = get_forex_price(forex_info)

            # Tambahkan informasi ke dalam list
            forex_list.append({
                'Nama': forex_name,
                'Kode': forex_symbol,
                'Harga': forex_price,
            })

    return pd.DataFrame(forex_list)

def get_forex_price(forex_info):
    try:
        # Mendapatkan harga forex dari yfinance
        forex_data = forex_info.history(period="1d")
        if not forex_data.empty:
            return forex_data['Close'][0]
    except Exception as e:
        print(f"Error getting price for {forex_info}: {e}")
    return None

def main():
    st.title('Tabel Pasangan Mata Uang FOREX📊')
    st.write("FOREX atau Foreign Exchange adalah pasar global untuk perdagangan mata uang. Ini adalah pasar terbesar dan paling likuid di dunia, di mana triliunan dolar diperdagangkan setiap hari. Pasangan mata uang utama seperti EUR/USD, GBP/USD, dan USD/JPY adalah beberapa yang paling populer di pasar ini. Berikut adalah beberapa pasangan mata uang yang populer dan nilai tukarnya saat ini:")

    # Pasangan mata uang yang ingin ditampilkan dalam tabel
    pairs = ['JPY=X', 'EURUSD=X', 'AUDUSD=X', "IDR=X", 'HKD=X', 'BNDUSD=X']

    # Mendapatkan data pasangan mata uang
    forex_df = get_forex_data(pairs)

    # Menampilkan tabel pasangan mata uang
    st.table(forex_df)
    
    st.write('Untuk informasi lebih lengkap, kunjungi Yahoo Finance https://finance.yahoo.com/')
    
if __name__ == '__main__':
    main()
