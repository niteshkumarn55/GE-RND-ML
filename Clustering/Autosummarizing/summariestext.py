from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.probability import FreqDist
from collections import defaultdict
from nltk.corpus import stopwords
from string import punctuation
from heapq import nlargest

_text = "Using the Goal Navigator. The goal navigator guides you through high-level goals that you might want to accomplish in vRealize Automation. The goals you can achieve depend on your role. " \
        "To complete each goal, you must complete a sequence of steps that are presented on separate pages in the vRealize Automation console. " \
        "The goal navigator can answer the following questions: Where do I start? " \
        "What are all the steps I need to complete to achieve a goal? What are the prerequisites for completing a particular task? " \
        "Why do I need to do this step and how does this step help me achieve my goal? The goal navigator is hidden by default. You can expand the goal navigator by clicking the icon on the left side of the screen. " \
        "After you select a goal, you navigate between the pages needed to accomplish the goal by clicking each step. The goal navigator does not validate that you completed a step, or force you to complete steps in a particular order. " \
        "The steps are listed in the recommended sequence. You can return to each goal as many times as needed. " \
        "For each step, the goal navigator provides a description of the task you need to perform on the corresponding page. The goal navigator does not provide detailed information such as how to complete the forms on a page. " \
        "You can hide the page information or move it to a more convenient position on the page. If you hide the page information, you can display it again by clicking the information icon on the goal navigator panel."

_text1 = "Ant Financial Services Group, the Chinese fintech giant, is planning a funding round that could fetch a valuation similar to the world’s biggest and oldest banks: The digital payment company is raising as much as $5 billion in equity that may value it at more than $100 billion, according to Reuters. "\
"Comparing market capitalizations of public companies with a private company’s valuation is an imperfect exercise. But in the broadest sense, and accepting these limitations, it shows investors think prospects for next-generation companies like Ant Financial are comparable to top-tier institutions like Goldman Sachs, which had a $94 billion market cap at the time of writing. "\
"There are several reasons investors are betting big on fintech upstarts. Chinese firms like Ant Financial and Tencent are unencumbered by the legacy of bank branches and aging technology that older banks elsewhere are trying to streamline. These younger companies also benefit from lower regulatory hurdles and have demographics on their side, thanks to mobile-friendly, millennial-heavy home markets; China has some 520 million mobile payment users. "\
"Ant’s latest fundraising could help position it for an initial public offering. The company was set up in 2004 (as Alipay) to provide PayPal-style payment services for online shopping. It was spun off (pdf) from Jack Ma’s Alibaba about a decade later (as Ant Financial). Alibaba recently said it will take a 33% stake in Ant, replacing the current system in which Alibaba receives royalties from Ant. "\
"Ant’s Alipay unit is now China’s biggest online payment platform. It also has stakes in Indian mobile-payment firm Paytm (paywall) and Thailand’s Ascend Money. "\
"While payments aren’t necessarily a cash cow (paywall), the service is a way to acquire customers and then sell them a range of more lucrative services: Ant provides services such as wealth management, lending, credit scoring, and insurance. If investors are right, Ant Financial could soon become a major force in the world of finance. "


_text2 = "In 2010, India-born Vikram Rangnekar took up a job as a software engineer at professional networking portal LinkedIn. Along with his family, he moved into a house at the foot of the Santa Cruz mountains in Silicon Valley. Earlier, Rangnekar had built an enterprise collaboration startup in Singapore called Socialwok, which had won the DemoPit competition at TechCrunch50, a showcase for startups. " \
"But just as everything seemed to be going well for Rangnekar—he was awaiting his green card—the family traded in its American dream and moved to Toronto, Canada." \
"I loved my time at LinkedIn and our life in California, but I couldn’t see myself spending my most productive years on a restrictive visa, Rangnekar told Quartz. Canada offered us a PR (permanent residency) and me a chance to go build a startup. " \
"After moving, Rangnekar founded Webmatr, a server-less app platform for developers, and launched a cloud-based organiser Bell+Cat. Before that, he also published a book, How to Build the Future, on how smart companies like Snapchat, Spotify, and others are using the Google Cloud. " \
"Rangnekar isn’t alone in making a move northwards from the US. " \
"With the Donald Trump administration mounting pressure on immigrants, several other Indian professionals in the US are thinking along the same lines. " \
"The American horror " \
"Trump’s protectionist stance has left Indians working in the US frightened and insecure. " \
"Last March, his government temporarily halted the premium processing of the H-1B, which involved clearing visa applications within 15 days for an additional fee—the standard procedure takes between three and six months. A month later, authorities turned up the heat on computer programmers by making the criteria to qualify for the H-1B harder. Earlier this month, the rules for third-party contract work were made more stringent, shortening the tenure of the visas and toughening renewal clauses. " \
"Certainty and security of their immigration status in the US, which is very closely tied to how successful they will be professionally and personally, is the biggest problem that Indian immigrants in the US are currently facing, Vivek Tandon, founder and CEO of EB5 BRICS, an advisory firm specialising in the half-a-million dollar investor route to the green card, told Quartz. Canada, in comparison, has far more liberal visa laws. " \
"Not only is the H-1B visa binding to one employer and the US citizenship hard to obtain, family members of immigrants are also at a loss. Most spouses of immigrant workers aren’t allowed to work in the US. Some of them—spouses of H-1B holders awaiting green cards—were given work authorisation a couple of years back and even that’s up for debate now. " \
"The effects of all these moves are beginning to show: Fewer Indian students enrolled in US universities in 2017 compared to 2016. And the homecoming of US-based Indians has spiked. " \
"Look, these are talented and in-demand people. If they decide the US isn’t a welcoming place, they’ll go to Canada, China, or Europe instead, Richard Burke, CEO of global immigration management platform Envoy, told Quartz. Talent is mobile, and US lawmakers need to understand that this country isn’t the only option for the world’s skilled workforce. " \
"While the US has increased visa scrutiny, Canada has only become friendlier. " \

"Between 2016 and 2017, the target for Canada’s economic class visas for skilled workers—the category most used by Indian immigrants—was hiked up from 160,600 to 172,500. " \

"Last year, Canadian prime minister Justin Trudeau announced the fast-track Global Skills Strategy programme which processes applications for those employed in tech occupations in a short span of two weeks. By comparison, the US Citizenship and Immigration Services (USCIS) takes between six and seven months, or longer, to approve the H-1Bs. " \

"Moreover, Canada’s Express Entry programme allows high-skilled talent in areas like software development, engineering, medical, and other academic professions to migrate to the country in under six months even without securing a job. The point-based system awards people for their level of education, work experience, and language ability in either English or French. Many H-1B workers in the US will be ideal candidates for Express Entry, as H-1B workers will excel in all of these areas, immigration law firm Allen and Hodgman writes. " \

"Canada currently admits far more high-skilled workers than the US—one permanent skilled visa for every 409 Canadian residents in 2016, nearly six times more per capita than the US, Richard Burke, CEO of immigration platform Envoy Global, told Quartz. " \

"Personally I feel it’s game over for the masters-H-1B-green card path used by so many Indians to build a life in the US. The wait time for getting a green card is over 20 years. This makes it as good as never, Rangnekar said. " \

"And though Silicon Valley is still the hotbed of innovation, Canada is quickly getting there. " \

"The new land of opportunity " \
"The tech ecosystem in Canada has attracted the top firms from Silicon Valley. Amazon has been hiring software developers, engineers, and programmers in Toronto. Uber has posted artificial intelligence (AI) and computer vision roles, among other tech positions, in the same city, too. " \

"When it comes to innovation and upward mobility, the US remains the gold standard, said Burke of Envoy. That said, Canada is making great strides as a welcoming and favourable destination, particularly among technology companies where team location is perhaps less important than in industries like healthcare, manufacturing, or financial services. " \

"Canadian companies have witnessed an uptick in applicants. For instance, Ottawa-based e-commerce platform Shopify logged 40% more applicants from the US in the first quarter of 2017 than it did during an average quarter in 2016. Also, Toronto-based digital medical image company Figure 1 received twice the number of US-based applicants for a senior role posted in January 2017 compared to a similar posting from a year ago. " \

"Myriad startups in upcoming sectors like artificial intelligence (AI), Internet of Things (IoT), and cryptocurrency are throwing up exciting employment opportunities of their own in Canada. " \

"The one drawback is the possible pay-cut techies need to take to relocate. Canadian techies are paid substantially lower than their US counterparts. For instance, while techies in San Francisco rake in $134,000 annually, the average salary in Toronto was a much lower $74,000 (or $97,000 Canadian), as per employment website Hired. "\

"However, experts believe it’s hardly a trade-off for stability and peace of mind. " \

"Canada’s flexible immigration policies, cultural diversity, democratic values, career opportunities, and large communities of the Indian diaspora lure thousands of Indians to apply for a permanent residency visa, Poorvi Chothani, managing partner at immigration law firm LawQuest, said. The cost of living in Canadian cities is also comparatively more affordable than Silicon Valley and perks like free healthcare are a big draw. "


_text3= "The days of paying with cash or card could soon be behind us. As security fears subside, and the world’s appetite for new ways to pay is rising. So much so that Mastercard has reported a 145% increase in Europeans tapping their cards to pay over the last year. This pin-free boom has led banks and payment companies to explore new ways to pay, many of which are on display at this year’s Mobile World Congress (MWC) in Barcelona. "\
"Europe leads the world in contactless payments and its overwhelming success has created a demand for even greater convenience, explains Paolo Battiston, executive vice president Digital Payments & Labs Europe at Mastercard. Shoppers’ trust in contactless is greater than ever, and in turn it seems they are ready to take this one stage further by trying contactless through connected devices. "\
"While many of the methods on display are, unsurprisingly, tied to a smartphone, there are a number of devices that have been designed to be standalone, fitting discreetly into daily life. One trial, conducted by Dutch bank ABN AMRO, saw 500 customers pay for goods using contactless rings, bracelets, watches, and keyrings. "\
"Yvonne Duits, product owner at ABN AMRO, believes these devices will improve the payment experience; With customer expectations clear and the new technology available today, the time has come to drop cumbersome methods of payment and embrace a better consumer experience through wearable payments. "\
"Another partnership announced at MWC is looking to change the way we pay for fuel or parking. Using the SAP Vehicle Network–a platform for connected cars–Mastercard plan to offer banks and retailers the tech to allow your car to act as your card, rapidly speeding up time at fuel stations, charging points, and parking spaces. "\
"While Europe may be leading the way with contactless payments, the story in Africa is very different. With a vast unbanked population and lack of infrastructure, mobile transactions dominate. "\
"With many essentials we take for granted simply not available, payment solutions have to be different. Around 625 million people lack access to electricity, instead relying on kerosene, candles and disposable batteries. M-KOPA is changing that, having connected over 600,000 African homes to pay-as-you-go solar power and appliances. Mastercard and M-KOPA are now exploring new ways to conveniently collect payments from off grid, low income homes in Africa and around the world. "\
"We may take for granted our ability to produce light with the simple flick of a switch. But for many around the world, simple things like having electricity can be life changing,” said Kiki Del Valle, senior vice president, Commerce for Every Device, Mastercard. By using our digital payment capabilities, we want to make it easy for people to access reliable and regular sources of energy and become more economically resilient. "\
"Using Mastercard’s Masterpass QR–a QR code that allows for payments–customers buy a solar system on credit and use their phone to make small daily payments for less than what they previously spent on hazardous, kerosene lamps. After paying for their system for roughly a year, customers build a credit rating that can be used to buy other products, such as solar-powered TVs and energy-efficient stoves. "\
"Masterpass QR will also be making its social debut in Africa later this year, as part of a partnership with Facebook. The trial in Nigeria will feature a Messenger bot that allows small businesses to easily set up and accept cash-free payments using QR codes. "\
"Unlike lots of the technology shown at trade shows, these examples are coming to market in the near future, potentially rendering physical cash a thing of the past and your wallet nothing more than a portable photo album. "\

_stopwords = set(stopwords.words('english')+list(punctuation))


def auto_summary_of_text(text,n):
    """

    :param text: Give the text that needs to be auto summarized
    :param n: specify the number of sentence thats needs to be shown after the auto summarization process is done
    :return: Prints the text in the console
    """
    autoSummarizedText = ""
    # Tokenizes the sentences from the text thats been passed
    sentences = sent_tokenize(text)
    # checks whether the text contain minimum number of the sentence that you asked to display by entering the number n
    assert n <= len(sentences)
    words_in_text = word_tokenize(text.lower())
    # removes the unneccsary words for the process to continue
    words = [word for word in words_in_text if word not in _stopwords]
    # Get the freq of most occured value added words in the sentence
    freq = FreqDist(words)
    # Get the most important words based on the text thats given
    nlargest(10,freq,key=freq.get)
    # Ranking the most important sentences that needs to be displayed as an abstract for the user
    ranking = defaultdict(int)
    for i, sent in enumerate(sentences):
        for w in word_tokenize(sent.lower()):
            if w in freq:
                ranking[i] += freq[w]

    # Sorting the sentences that's ranked in order to display the initail sentence from the text first
    sents_id = nlargest(n,ranking,key=ranking.get)
    for i in sorted(sents_id):
        if autoSummarizedText=="":
            autoSummarizedText = sentences[i]
        else:
            autoSummarizedText = autoSummarizedText +" "+ sentences[i]

    return autoSummarizedText

if __name__ == '__main__':
    print(auto_summary_of_text(_text3,4))