from bs4 import BeautifulSoup
import urllib.request
import json


if __name__ == '__main__':
    save_file = "data/raw/courses_new.json"
    mooc_list_domain = "https://www.mooc-list.com"
    start_page = 0
    end_page = 10000
    url_template = "https://www.mooc-list.com/multiple-criteria?&field_university_entity_tid=&&&&&&&field_peer_assessments_value=All&field_team_projects_value=All&&&field_coupon_value=All&field_audio_lectures_value=All&field_video_lectures_value=All&field_tags_tid=&field_start_date_value_op=between&field_start_date_value[value]&field_start_date_value[min]&field_start_date_value[max]&sort_by=field_rating_rating&sort_order=DESC&page="
    header = {'User-Agent': 'Mozilla/5.0'}
    urls_to_parse = (url_template + str(page_number) for page_number in range(start_page, end_page))
    data = []
    with open(save_file, "w") as out_file:
        for c, url in enumerate(urls_to_parse):
            print("Parsing %d url" % c)
            req = urllib.request.Request(url, headers=header)
            with urllib.request.urlopen(req) as response:
                html_doc = response.read()
                soup = BeautifulSoup(html_doc, 'html.parser')
                if soup.find("div", "view-empty"):
                    break
                for course_desc in soup.find_all("div", "views-row"):
                    in_course_divs = course_desc.find_all("div", "views-field")
                    mooc_list_url = mooc_list_domain + in_course_divs[0].span.a["href"]

                    course_req = urllib.request.Request(mooc_list_url, headers=header)
                    with urllib.request.urlopen(course_req) as course_response:
                        course_doc = course_response.read()
                        course_soup = BeautifulSoup(course_doc, 'html.parser')
                        instr_divs = course_soup.find_all("div", "title-info")[1].div.div
                        instructors = [instructor.a.text for instructor in instr_divs.find_all("div")]
                        text = None
                        try:
                            text = course_soup.find(id="corpoCurso").div.div.div.p.text
                        except AttributeError:
                            pass

                    title = in_course_divs[0].span.a.text
                    start_date = in_course_divs[1].span.text
                    university_url = mooc_list_domain + in_course_divs[2].div.a["href"]
                    university = in_course_divs[2].div.a.text
                    provider = in_course_divs[3].div.a.text
                    language = in_course_divs[4].div.a.text
                    subtitles = None
                    try:
                        subtitles = in_course_divs[5].div.a.text
                    except AttributeError:
                        pass

                    short_description = None
                    try:
                        short_description = in_course_divs[6].div.p.text
                    except AttributeError:
                        pass
                    fivestar_summary = in_course_divs[7].find("div", "fivestar-summary").find_all("span")
                    average_rating = None
                    votes = None
                    try:
                        average_rating = fivestar_summary[1].text
                        votes = fivestar_summary[3].text
                    except IndexError:
                        pass
                    except AttributeError:
                        pass
                    categories = [category.text for category in in_course_divs[8].div.find_all("a")]
                    tags = [tag.text for tag in in_course_divs[9].div.find_all("a")]

                    result = {"title": title,
                              "instructors": "; ".join(instructors),
                              "text": text,
                              "mooc_list_url": mooc_list_url,
                              "start_date": start_date,
                              "university_url": university_url,
                              "university": university,
                              "provider": provider,
                              "language": language,
                              "subtitles": subtitles if subtitles else "None",
                              "short_description": short_description if short_description else "None",
                              "average_rating": average_rating if average_rating else "None",
                              "votes": votes if votes else "None",
                              "categories": "; ".join(categories),
                              "tags": "; ".join(tags) if tags is not None else "None"}

                    data.append(result)
        json.dump(data, out_file)
