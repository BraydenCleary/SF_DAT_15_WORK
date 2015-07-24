require 'foursquare2'
require 'yaml'
require 'csv'

PATH_TO_FOURSQUARE = "./data/checkin_data.csv"
CREDENTIALS = YAML.load(File.read('./foursquare_credentials.yml'))

PARK_VENUES = {
  alamo_square: '4460d38bf964a5200a331fe3',
  golden_gate: '445e36bff964a520fb321fe3',
  crissy: '40b7d280f964a52093001fe3',
  fort_mason: '4bae72d5f964a520bdb33be3',
  dolores: '4ab595e1f964a520877520e3',
  washington_square: '4486b2d2f964a5202d341fe3'
}

client = Foursquare2::Client.new(:api_version => '20140806', client_secret: CREDENTIALS['client_secret'], client_id: CREDENTIALS['client_id'])

CSV.open(PATH_TO_FOURSQUARE, 'ab') do |csv|
  PARK_VENUES.each do |park_name, park_id|
    csv << [park_id, client.venue(park_id).values[7].checkinsCount, park_name, Time.now]
    sleep(86400)
  end
end
