import os

import streamlit as st


def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def memo_ui(text_area_key):
    st.markdown("###### 📝 予想メモ機能")

    if text_area_key not in st.session_state:
        st.session_state[text_area_key] = ""

    memo_cols = st.columns([3, 1, 1])
    with memo_cols[0]:
        memo_options = ["---"] + st.session_state.saved_memos
        selected_memo = st.selectbox(
            "保存したメモ",
            memo_options,
            key=f"{text_area_key}_memo_selector",
            label_visibility="collapsed",
        )

    with memo_cols[1]:
        if st.button("読み込む", key=f"{text_area_key}_load_memo", use_container_width=True):
            if selected_memo != "---":
                st.session_state[text_area_key] = selected_memo
                st.rerun()

    with memo_cols[2]:
        if st.button("削除", key=f"{text_area_key}_delete_memo", use_container_width=True):
            if selected_memo != "---" and selected_memo in st.session_state.saved_memos:
                st.session_state.saved_memos.remove(selected_memo)
                st.toast("選択したメモを削除しました。")
                st.rerun()

    if st.button("現在の内容をメモとして保存", key=f"{text_area_key}_save_memo_main", use_container_width=True):
        current_text = st.session_state.get(text_area_key, "")
        if current_text and current_text not in st.session_state.saved_memos:
            st.session_state.saved_memos.append(current_text)
            st.toast("メモを保存しました。")
            st.rerun()
        elif not current_text:
            st.toast("保存する内容がありません。")
        else:
            st.toast("このメモは既に保存されています。")
