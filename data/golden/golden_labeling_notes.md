# Golden Set Labeling Notes

Total examples: 220
Corrected on human review: 116 (52.7%)

## Final intent distribution

intent
device_bug               54
software_update_issue    46
other_unclear            41
battery_performance      21
howto_info               16
account_security         16
order_shipping           12
billing_refund            9
praise_feedback           5

## Final escalate distribution

escalate
no     138
yes     82

## Sample corrections (draft -> final)

- "This doesn't help either... Unable to trace why I'm charged. Please help"
    draft=battery_performance -> final=billing_refund  [charge complaint, not battery (keyword false-matched "charg" in "charged")]
- "Not buying that bullshit iPhone x so y’all can stop fucking with my phone and stop freezin"
    draft=device_bug -> final=device_bug  [extreme anger/profanity - escalate for de-escalation despite known bug pattern]
- "Hey , you think it’s about time to sort out your glitchy new ios?"
    draft=device_bug -> final=software_update_issue  [explicitly "new ios" - grouped with update-tagged bugs]
- "iPhone 6s Plus, ios 11.1.1 and it is set to “set automatically”"
    draft=software_update_issue -> final=other_unclear  [mid-thread fragment with no standalone complaint]
- "Bin ich der einzige, bei dem die #Mail App seit dem letzten #iOS #Update heftig hängt? #iP"
    draft=software_update_issue -> final=software_update_issue  [non-English (German) - forcing escalate; translation risk]
- "can you please update your maps for the new queensferry bridge. I’m nae in the middle of t"
    draft=software_update_issue -> final=howto_info  [Maps data feature request, not an update failure]
- "Hi when sending an iMessage it tried to get through but it comes up with a not delivered m"
    draft=order_shipping -> final=device_bug  [iMessage delivery failure - keyword false-matched "delivered"]
- "Alright I can’t figure this one out. Downloaded a purchased album to my new X, but it’s ou"
    draft=order_shipping -> final=device_bug  [music track order bug in app - keyword false-matched "order"]
- "look I’ve taken all the updates but why the phone glitchy and I have to press things 5x in"
    draft=order_shipping -> final=device_bug  [keyword false-matched "in order for" - actual issue is UI glitch]
- "is blaming , and vice versa. Users are quitting Safari over this in OS High Sierra. Someon"
    draft=order_shipping -> final=software_update_issue  [Safari/High Sierra bug, no real order-keyword basis]
- "Dear, Y’all really got me fucked up. A whole update and i still can’t see the letter I. I "
    draft=billing_refund -> final=device_bug  ["I" autocorrect bug rant; refund/money is venting, not the ask]
- "#iphone7plus and the camera doesn’t work anymore #refund #badupdate"
    draft=billing_refund -> final=device_bug  [camera broken after update; #refund is a hashtag not the core ask]
- "Dutch ios 11 fail.. This is the choice btwn ‘busy’ and ‘free’. In agenda. Only translated "
    draft=billing_refund -> final=software_update_issue  [Dutch calendar translation bug after iOS 11]
- "we still doin this....I I️ I️ I️ I️ ? Y’all annoying! FIX IT. I️ PAY TOO MUCH MONEY FOR TH"
    draft=billing_refund -> final=device_bug  ["I" autocorrect bug rant; money mention incidental]
- "why did my phone go from 30% to 1% within a minute??? Y’all need to stop playing. I don’t "
    draft=billing_refund -> final=battery_performance  [battery drain complaint; "money" mention is sarcastic aside]
