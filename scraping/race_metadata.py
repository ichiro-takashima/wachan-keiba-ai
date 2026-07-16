"""db.netkeiba.com のレースページ(1レース分のHTML)からメタ情報を抽出する。

出馬情報26カラムのうち、結果テーブル(race_table_01)に含まれない
日付・開催・クラス・芝ダート・距離・回り・馬場・天気・場id・場名を担当する。
数値への変換は preprocessing/encode.py 側で行うため、ここではテキストのまま返す。
"""
import re

from bs4 import BeautifulSoup

PLACE_NAME_BY_ID = {
    "01": "札幌",
    "02": "函館",
    "03": "福島",
    "04": "新潟",
    "05": "東京",
    "06": "中山",
    "07": "中京",
    "08": "京都",
    "09": "阪神",
    "10": "小倉",
}

_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)


def clean_html_comments(html_text):
    """h1タグ内などに紛れ込むHTMLコメントを除去する(bs4のget_text()はコメントも拾うため)。"""
    return _COMMENT_PATTERN.sub("", html_text)


def _split_span_parts(span_text):
    normalized = span_text.replace("\xa0", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return [part.strip() for part in normalized.split("/") if part.strip()]


def _parse_course_part(part):
    # 「芝右 外3200m」のように回りと距離の間に外/内(コース変則)が挟まる場合や、
    # 障害レースの「障芝3380m」のように障害＋本来のコース種別が続く場合がある。
    match = re.match(r"(障害|障)?\s*(芝|ダート|ダ)?\s*(左|右|直)?\s*(?:外|内)?\s*(\d+)m", part)
    if not match or (not match.group(1) and not match.group(2)):
        return None, None, None
    if match.group(1):
        surface = "障害"
    else:
        surface = "ダート" if match.group(2) in ("ダ", "ダート") else match.group(2)
    turn = match.group(3) or ""
    distance = float(match.group(4))
    return surface, turn, distance


def parse_race_metadata(html_text, race_id):
    """レースページのHTML全文とrace_idから、メタ情報の辞書を返す。取得できない項目は空文字/NAとする。"""
    text = clean_html_comments(html_text)
    soup = BeautifulSoup(text, "html.parser")

    result = {
        "race_id": race_id,
        "レース名": "",
        "日付": "",
        "開催": "",
        "クラス": "",
        "芝・ダート": "",
        "距離": None,
        "回り": "",
        "馬場": "",
        "天気": "",
        "場id": race_id[4:6] if len(race_id) >= 6 else "",
    }
    result["場名"] = PLACE_NAME_BY_ID.get(result["場id"], "")

    dl = soup.find("dl", class_="racedata")
    class_text_parts = []

    if dl:
        h1 = dl.find("h1")
        if h1:
            race_name = h1.get_text(" ", strip=True)
            result["レース名"] = race_name
            grade_match = re.search(r"\((G[I]{1,3}|Ｇ[I]{1,3}|L)\)", race_name)
            if grade_match:
                class_text_parts.append(grade_match.group(0))

        span = dl.find("span")
        if span:
            for part in _split_span_parts(span.get_text(" ", strip=True)):
                # 1つ目: "芝左1600m" のようなコース情報(距離を含む)
                surface, turn, distance = _parse_course_part(part)
                if surface is not None:
                    result["芝・ダート"] = surface
                    result["回り"] = turn
                    result["距離"] = distance
                    continue
                # 2つ目: "芝 : 良" / "ダート : 稍重" のような馬場状態(距離を含まない)
                going_match = re.match(r"(?:芝|ダート|ダ)\s*[:：]\s*(\S+)", part)
                if going_match:
                    result["馬場"] = going_match.group(1)
                    continue
                weather_match = re.match(r"天候\s*[:：]\s*(\S+)", part)
                if weather_match:
                    result["天気"] = weather_match.group(1)

    smalltxt = soup.find("p", class_="smalltxt")
    if smalltxt:
        smalltxt_text = re.sub(r"\s+", " ", smalltxt.get_text(" ", strip=True)).strip()

        date_match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", smalltxt_text)
        if date_match:
            y, m, d = date_match.groups()
            result["日付"] = f"{y}/{int(m):02d}/{int(d):02d}"

        meeting_match = re.search(r"(\d+)回(\S+?)(\d+)日目", smalltxt_text)
        if meeting_match:
            kai, place_name, day = meeting_match.groups()
            result["開催"] = f"{kai}回{place_name}{day}日目"
            if not result["場名"]:
                result["場名"] = place_name
            remainder = smalltxt_text[meeting_match.end():].strip()
        else:
            remainder = smalltxt_text

        if remainder:
            class_text_parts.append(remainder)

    result["クラス"] = " ".join(class_text_parts).strip()
    return result
