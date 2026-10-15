from cert_ua.entities import extract_entities


def values(r, t, key="normalized"):
    return [g[key] for g in r[t]]


def test_identifiers_normalised():
    text = "Кластер uac-0226 (CERT-UA #11689, CERT-UA#11689) експлуатує cve-2023-38831 та CVE-2017-0199."
    r = extract_entities(text)
    assert values(r, "uac_ids") == ["UAC-0226"]
    assert values(r, "cert_ua_ids") == ["CERT-UA#11689"]
    assert r["cert_ua_ids"][0]["variants"] == ["CERT-UA #11689", "CERT-UA#11689"]
    assert values(r, "cve_ids") == ["CVE-2023-38831", "CVE-2017-0199"]


def test_hashes_by_length():
    md5, sha1 = "4a2ec9f72b910c0a8e3efc4c334f5bad", "1d7c1ff5b361d823feb56ca134e026932884440e"
    sha256 = "f8a31715840852e8ef04016b31f909123f2aa864f3850c45eb511fd1885b4037"
    r = extract_entities(f"{md5}  {sha1}\n{sha256.upper()} {'a' * 32} {'1' * 32} {md5}ff")
    assert [(g["normalized"], g["hash_type"]) for g in r["file_hashes"]] == [
        (md5, "md5"), (sha1, "sha1"), (sha256, "sha256")]


def test_defanged_network_indicators():
    text = ("hXXps://drobbox[.]cloud/x.exe (tcp)://37[.]120.239.187:6501 http[:]//1833[.]site/a[.]exe "
            "149[.]102.246.110 Chrome/111.0.0.0 ver 1.2.3.4.5 (https://www.drupal.org/psa-2018-003).")
    r = extract_entities(text)
    assert values(r, "urls") == ["https://drobbox.cloud/x.exe", "tcp://37.120.239.187:6501",
                                 "http://1833.site/a.exe", "https://www.drupal.org/psa-2018-003"]
    assert values(r, "urls", "value")[0] == "hXXps://drobbox[.]cloud/x.exe"
    assert values(r, "ipv4_addresses") == ["37.120.239.187", "149.102.246.110"]
    assert "drobbox.cloud" in values(r, "domains") and "1833.site" in values(r, "domains")


def test_domain_false_positives_rejected():
    text = ("(new-object System.Net.WebClient).downloadfile, %TMP%\\status.zip, RuntimeBroker.exe, "
            "run.sh, т.д., Compress-Archive; legit: evil[.]com, example.org, mail.gov.ua")
    r = extract_entities(text)
    assert values(r, "domains") == ["evil.com", "example.org", "mail.gov.ua"]


def test_offsets_and_sections():
    body = "Вступ\n\nМережеві:\n\nC2: 89[.]44.9.186 та evil[.]com"
    sections = [{"heading": None, "start_char": 0}, {"heading": "Мережеві:", "start_char": 7}]
    r = extract_entities(body, sections, title="UAC-0001 атакує")
    ip = r["ipv4_addresses"][0]["mentions"][0]
    assert body[ip["start"]:ip["end"]] == "89[.]44.9.186" and ip["section"] == "Мережеві:"
    uac = r["uac_ids"][0]["mentions"][0]
    assert uac["field"] == "title" and uac["start"] == 0
    assert r["counts"]["domains"] == 1


def test_emails():
    r = extract_entities("Пишіть на cert@cert.gov.ua або antiteror2[@]ukr[.]net")
    assert values(r, "email_addresses") == ["cert@cert.gov.ua", "antiteror2@ukr.net"]
