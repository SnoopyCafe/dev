import com.pluralsight.service.CustomerService;
import com.pluralsight.service.CustomerServiceImpl;

public class Application {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		CustomerService service = new CustomerServiceImpl();
		System.out.println(service.findAll().get(1).getFirstname());
		
	}

}
